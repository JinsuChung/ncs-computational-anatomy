CREATE SCHEMA IF NOT EXISTS ncs;

CREATE TABLE IF NOT EXISTS ncs.import_batches (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_file TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    row_count BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ncs.raw_rows (
    batch_id BIGINT NOT NULL REFERENCES ncs.import_batches(id) ON DELETE CASCADE,
    sheet_name TEXT NOT NULL,
    row_number INTEGER NOT NULL,
    major_category_code TEXT NOT NULL,
    major_category_name TEXT NOT NULL,
    middle_category_code TEXT NOT NULL,
    middle_category_name TEXT NOT NULL,
    minor_category_code TEXT NOT NULL,
    minor_category_name TEXT NOT NULL,
    detail_category_code TEXT NOT NULL,
    detail_category_name TEXT NOT NULL,
    competency_unit_code TEXT NOT NULL,
    competency_unit_name TEXT NOT NULL,
    competency_unit_level INTEGER,
    competency_unit_element_code TEXT NOT NULL,
    competency_unit_element_name TEXT NOT NULL,
    competency_unit_element_level INTEGER,
    performance_criterion_no TEXT NOT NULL,
    performance_criterion_description TEXT NOT NULL,
    kta_type_code TEXT NOT NULL,
    kta_type_name TEXT NOT NULL,
    kta_item_no TEXT NOT NULL,
    kta_item_description TEXT NOT NULL,
    PRIMARY KEY (batch_id, sheet_name, row_number)
);

CREATE TABLE IF NOT EXISTS ncs.major_categories (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ncs.middle_categories (
    full_code TEXT PRIMARY KEY,
    major_category_code TEXT NOT NULL REFERENCES ncs.major_categories(code),
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE (major_category_code, code)
);

CREATE TABLE IF NOT EXISTS ncs.minor_categories (
    full_code TEXT PRIMARY KEY,
    middle_category_full_code TEXT NOT NULL REFERENCES ncs.middle_categories(full_code),
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE (middle_category_full_code, code)
);

CREATE TABLE IF NOT EXISTS ncs.detail_categories (
    full_code TEXT PRIMARY KEY,
    minor_category_full_code TEXT NOT NULL REFERENCES ncs.minor_categories(full_code),
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE (minor_category_full_code, code)
);

CREATE TABLE IF NOT EXISTS ncs.competency_units (
    code TEXT PRIMARY KEY,
    detail_category_full_code TEXT NOT NULL REFERENCES ncs.detail_categories(full_code),
    name TEXT NOT NULL,
    level INTEGER
);

CREATE TABLE IF NOT EXISTS ncs.competency_unit_elements (
    code TEXT PRIMARY KEY,
    competency_unit_code TEXT NOT NULL REFERENCES ncs.competency_units(code),
    element_no TEXT NOT NULL,
    name TEXT NOT NULL,
    level INTEGER
);

CREATE TABLE IF NOT EXISTS ncs.performance_criteria (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    competency_unit_element_code TEXT NOT NULL REFERENCES ncs.competency_unit_elements(code),
    criterion_no TEXT NOT NULL,
    description TEXT NOT NULL,
    UNIQUE (competency_unit_element_code, criterion_no, description)
);

CREATE TABLE IF NOT EXISTS ncs.kta_types (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ncs.kta_items (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    performance_criterion_id BIGINT NOT NULL REFERENCES ncs.performance_criteria(id),
    kta_type_code TEXT NOT NULL REFERENCES ncs.kta_types(code),
    item_no TEXT NOT NULL,
    description TEXT NOT NULL,
    UNIQUE (performance_criterion_id, kta_type_code, item_no, description)
);

CREATE INDEX IF NOT EXISTS idx_raw_rows_batch_id ON ncs.raw_rows (batch_id);
CREATE INDEX IF NOT EXISTS idx_raw_rows_competency_unit_code ON ncs.raw_rows (competency_unit_code);
CREATE INDEX IF NOT EXISTS idx_middle_categories_major ON ncs.middle_categories (major_category_code);
CREATE INDEX IF NOT EXISTS idx_minor_categories_middle ON ncs.minor_categories (middle_category_full_code);
CREATE INDEX IF NOT EXISTS idx_detail_categories_minor ON ncs.detail_categories (minor_category_full_code);
CREATE INDEX IF NOT EXISTS idx_competency_units_detail ON ncs.competency_units (detail_category_full_code);
CREATE INDEX IF NOT EXISTS idx_competency_unit_elements_unit ON ncs.competency_unit_elements (competency_unit_code);
CREATE INDEX IF NOT EXISTS idx_performance_criteria_element ON ncs.performance_criteria (competency_unit_element_code);
CREATE INDEX IF NOT EXISTS idx_kta_items_criterion ON ncs.kta_items (performance_criterion_id);
CREATE INDEX IF NOT EXISTS idx_kta_items_type ON ncs.kta_items (kta_type_code);

CREATE OR REPLACE VIEW ncs.classification_paths AS
SELECT
    major.code AS major_category_code,
    major.name AS major_category_name,
    middle.code AS middle_category_code,
    middle.name AS middle_category_name,
    minor.code AS minor_category_code,
    minor.name AS minor_category_name,
    detail.code AS detail_category_code,
    detail.name AS detail_category_name,
    detail.full_code AS detail_category_full_code
FROM ncs.detail_categories detail
JOIN ncs.minor_categories minor ON minor.full_code = detail.minor_category_full_code
JOIN ncs.middle_categories middle ON middle.full_code = minor.middle_category_full_code
JOIN ncs.major_categories major ON major.code = middle.major_category_code;

CREATE OR REPLACE VIEW ncs.ontology_edges AS
SELECT major.code AS subject_code, 'hasMiddleCategory' AS predicate, middle.full_code AS object_code
FROM ncs.major_categories major
JOIN ncs.middle_categories middle ON middle.major_category_code = major.code
UNION ALL
SELECT middle.full_code, 'hasMinorCategory', minor.full_code
FROM ncs.middle_categories middle
JOIN ncs.minor_categories minor ON minor.middle_category_full_code = middle.full_code
UNION ALL
SELECT minor.full_code, 'hasDetailCategory', detail.full_code
FROM ncs.minor_categories minor
JOIN ncs.detail_categories detail ON detail.minor_category_full_code = minor.full_code
UNION ALL
SELECT detail.full_code, 'hasCompetencyUnit', unit.code
FROM ncs.detail_categories detail
JOIN ncs.competency_units unit ON unit.detail_category_full_code = detail.full_code
UNION ALL
SELECT unit.code, 'hasElement', element.code
FROM ncs.competency_units unit
JOIN ncs.competency_unit_elements element ON element.competency_unit_code = unit.code
UNION ALL
SELECT element.code, 'hasPerformanceCriterion', criterion.id::TEXT
FROM ncs.competency_unit_elements element
JOIN ncs.performance_criteria criterion ON criterion.competency_unit_element_code = element.code
UNION ALL
SELECT criterion.id::TEXT, 'requiresKnowledgeSkillAttitude', item.id::TEXT
FROM ncs.performance_criteria criterion
JOIN ncs.kta_items item ON item.performance_criterion_id = criterion.id;
