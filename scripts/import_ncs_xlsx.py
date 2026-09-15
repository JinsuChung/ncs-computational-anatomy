#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable

import openpyxl
import psycopg


EXPECTED_HEADERS = [
    "대분류코드",
    "대분류코드명",
    "중분류코드",
    "중분류코드명",
    "소분류코드",
    "소분류코드명",
    "세분류코드",
    "세분류코드명",
    "능력단위분류번호",
    "능력단위명칭",
    "수준",
    "능력단위요소번호",
    "능력단위요소명",
    "능력단위요소수준",
    "수행준거번호",
    "수행준거",
    "지식기술태도코드",
    "지식기술태도코드명",
    "지식기술태도번호",
    "지식기술태도의의",
]

RAW_COLUMNS = [
    "batch_id",
    "sheet_name",
    "row_number",
    "major_category_code",
    "major_category_name",
    "middle_category_code",
    "middle_category_name",
    "minor_category_code",
    "minor_category_name",
    "detail_category_code",
    "detail_category_name",
    "competency_unit_code",
    "competency_unit_name",
    "competency_unit_level",
    "competency_unit_element_code",
    "competency_unit_element_name",
    "competency_unit_element_level",
    "performance_criterion_no",
    "performance_criterion_description",
    "kta_type_code",
    "kta_type_name",
    "kta_item_no",
    "kta_item_description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import NCS XLSX data into a normalized PostgreSQL schema."
    )
    parser.add_argument(
        "--xlsx",
        default="reference/NCS정보망DB(대분류별,2026년2월).xlsx",
        help="Path to the NCS XLSX file.",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"),
        help="PostgreSQL DSN. Defaults to DATABASE_URL or local docker-compose DB.",
    )
    parser.add_argument(
        "--schema-sql",
        default="sql/schema.sql",
        help="SQL file that creates the target schema.",
    )
    return parser.parse_args()


def resolve_xlsx_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    matches = sorted(Path("reference").glob("NCS*.xlsx"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def clean_int(value: object) -> int | None:
    text = clean(value)
    if text is None:
        return None
    return int(float(text))


def rows_from_workbook(path: Path, batch_id: int) -> Iterable[list[object]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for worksheet in workbook.worksheets:
            rows = worksheet.iter_rows(values_only=True)
            header = [clean(value) for value in next(rows)]
            if header != EXPECTED_HEADERS:
                raise ValueError(
                    f"Unexpected header in sheet {worksheet.title}: {header}"
                )

            for row_number, row in enumerate(rows, start=2):
                values = list(row[: len(EXPECTED_HEADERS)])
                if not any(clean(value) for value in values):
                    continue

                major_code = clean(values[0])
                major_name = clean(values[1])
                middle_code = clean(values[2])
                middle_name = clean(values[3])
                minor_code = clean(values[4])
                minor_name = clean(values[5])
                detail_code = clean(values[6])
                detail_name = clean(values[7])
                unit_code = clean(values[8])
                unit_name = clean(values[9])
                unit_level = clean_int(values[10])
                unit_element_code = clean(values[11])
                unit_element_name = clean(values[12])
                unit_element_level = clean_int(values[13])
                criterion_no = clean(values[14])
                criterion_description = clean(values[15])
                kta_type_code = clean(values[16])
                kta_type_name = clean(values[17])
                kta_item_no = clean(values[18])
                kta_item_description = clean(values[19])

                required_values = [
                    major_code,
                    major_name,
                    middle_code,
                    middle_name,
                    minor_code,
                    minor_name,
                    detail_code,
                    detail_name,
                    unit_code,
                    unit_name,
                    unit_element_code,
                    unit_element_name,
                    criterion_no,
                    criterion_description,
                    kta_type_code,
                    kta_type_name,
                    kta_item_no,
                    kta_item_description,
                ]
                if any(value is None for value in required_values):
                    raise ValueError(
                        f"Required value is empty in sheet {worksheet.title}, row {row_number}"
                    )

                yield [
                    batch_id,
                    worksheet.title,
                    row_number,
                    major_code,
                    major_name,
                    middle_code,
                    middle_name,
                    minor_code,
                    minor_name,
                    detail_code,
                    detail_name,
                    unit_code,
                    unit_name,
                    unit_level,
                    unit_element_code,
                    unit_element_name,
                    unit_element_level,
                    criterion_no,
                    criterion_description,
                    kta_type_code,
                    kta_type_name,
                    kta_item_no,
                    kta_item_description,
                ]
    finally:
        workbook.close()


def copy_raw_rows(conn: psycopg.Connection, rows: Iterable[list[object]]) -> int:
    count = 0

    with conn.cursor() as cursor:
        with cursor.copy(
            f"COPY ncs.raw_rows ({', '.join(RAW_COLUMNS)}) FROM STDIN"
        ) as copy:
            for row in rows:
                copy.write_row(row)
                count += 1

                if count % 100_000 == 0:
                    print(f"copied {count:,} rows...")

    return count


def transform_raw_rows(conn: psycopg.Connection, batch_id: int) -> None:
    statements = [
        """
        INSERT INTO ncs.major_categories (code, name)
        SELECT major_category_code, max(major_category_name)
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        GROUP BY major_category_code
        ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
        """,
        """
        INSERT INTO ncs.middle_categories (full_code, major_category_code, code, name)
        SELECT
            major_category_code || middle_category_code,
            major_category_code,
            middle_category_code,
            max(middle_category_name)
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        GROUP BY major_category_code, middle_category_code
        ON CONFLICT (full_code) DO UPDATE SET
            major_category_code = EXCLUDED.major_category_code,
            code = EXCLUDED.code,
            name = EXCLUDED.name
        """,
        """
        INSERT INTO ncs.minor_categories (full_code, middle_category_full_code, code, name)
        SELECT
            major_category_code || middle_category_code || minor_category_code,
            major_category_code || middle_category_code,
            minor_category_code,
            max(minor_category_name)
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        GROUP BY major_category_code, middle_category_code, minor_category_code
        ON CONFLICT (full_code) DO UPDATE SET
            middle_category_full_code = EXCLUDED.middle_category_full_code,
            code = EXCLUDED.code,
            name = EXCLUDED.name
        """,
        """
        INSERT INTO ncs.detail_categories (full_code, minor_category_full_code, code, name)
        SELECT
            major_category_code || middle_category_code || minor_category_code || detail_category_code,
            major_category_code || middle_category_code || minor_category_code,
            detail_category_code,
            max(detail_category_name)
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        GROUP BY major_category_code, middle_category_code, minor_category_code, detail_category_code
        ON CONFLICT (full_code) DO UPDATE SET
            minor_category_full_code = EXCLUDED.minor_category_full_code,
            code = EXCLUDED.code,
            name = EXCLUDED.name
        """,
        """
        INSERT INTO ncs.competency_units (code, detail_category_full_code, name, level)
        SELECT
            competency_unit_code,
            max(major_category_code || middle_category_code || minor_category_code || detail_category_code),
            max(competency_unit_name),
            max(competency_unit_level)
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        GROUP BY competency_unit_code
        ON CONFLICT (code) DO UPDATE SET
            detail_category_full_code = EXCLUDED.detail_category_full_code,
            name = EXCLUDED.name,
            level = EXCLUDED.level
        """,
        """
        INSERT INTO ncs.competency_unit_elements (
            code,
            competency_unit_code,
            element_no,
            name,
            level
        )
        SELECT
            competency_unit_element_code,
            max(competency_unit_code),
            regexp_replace(competency_unit_element_code, '^.* ', ''),
            max(competency_unit_element_name),
            max(competency_unit_element_level)
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        GROUP BY competency_unit_element_code
        ON CONFLICT (code) DO UPDATE SET
            competency_unit_code = EXCLUDED.competency_unit_code,
            element_no = EXCLUDED.element_no,
            name = EXCLUDED.name,
            level = EXCLUDED.level
        """,
        """
        INSERT INTO ncs.performance_criteria (
            competency_unit_element_code,
            criterion_no,
            description
        )
        SELECT DISTINCT
            competency_unit_element_code,
            performance_criterion_no,
            performance_criterion_description
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        ON CONFLICT (competency_unit_element_code, criterion_no, description) DO NOTHING
        """,
        """
        INSERT INTO ncs.kta_types (code, name)
        SELECT kta_type_code, max(kta_type_name)
        FROM ncs.raw_rows
        WHERE batch_id = %(batch_id)s
        GROUP BY kta_type_code
        ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
        """,
        """
        INSERT INTO ncs.kta_items (
            performance_criterion_id,
            kta_type_code,
            item_no,
            description
        )
        SELECT DISTINCT
            criterion.id,
            raw.kta_type_code,
            raw.kta_item_no,
            raw.kta_item_description
        FROM ncs.raw_rows raw
        JOIN ncs.performance_criteria criterion
            ON criterion.competency_unit_element_code = raw.competency_unit_element_code
            AND criterion.criterion_no = raw.performance_criterion_no
            AND criterion.description = raw.performance_criterion_description
        WHERE raw.batch_id = %(batch_id)s
        ON CONFLICT (performance_criterion_id, kta_type_code, item_no, description) DO NOTHING
        """,
    ]

    with conn.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement, {"batch_id": batch_id})


def main() -> None:
    args = parse_args()
    xlsx_path = resolve_xlsx_path(args.xlsx)
    schema_path = Path(args.schema_sql)

    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)

    source_sha256 = sha256_file(xlsx_path)

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute(schema_path.read_text(encoding="utf-8"))
            cursor.execute(
                """
                INSERT INTO ncs.import_batches (source_file, source_sha256)
                VALUES (%s, %s)
                RETURNING id
                """,
                (str(xlsx_path), source_sha256),
            )
            batch_id = cursor.fetchone()[0]

        print(f"created import batch {batch_id}")
        row_count = copy_raw_rows(conn, rows_from_workbook(xlsx_path, batch_id))
        print(f"copied {row_count:,} raw rows")

        transform_raw_rows(conn, batch_id)

        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ncs.import_batches
                SET finished_at = clock_timestamp(), row_count = %s
                WHERE id = %s
                """,
                (row_count, batch_id),
            )

    print("import completed")


if __name__ == "__main__":
    main()
