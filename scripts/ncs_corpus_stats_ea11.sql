\echo === kta_types / unit code sample
SELECT * FROM ncs.kta_types;
SELECT code, name, level FROM ncs.competency_units LIMIT 3;

\echo === [A5] 개정 역학: 대분류별 능력단위 코드의 연도(_YYvN) 분포
WITH u AS (
  SELECT cu.code, cp.major_category_name AS major,
         substring(cu.code from '_(\d\d)v(\d+)$') AS yy,
         (regexp_match(cu.code, '_(\d\d)v(\d+)$'))[2]::int AS ver
  FROM ncs.competency_units cu JOIN ncs.classification_paths cp ON cp.detail_category_full_code = cu.detail_category_full_code)
SELECT major, count(*) units,
       round(100.0*avg((yy::int >= 22)::int),1) AS pct_rev_2022plus,
       round(avg(yy::int),1) AS mean_yy, round(avg(ver),2) AS mean_ver, max(ver) AS max_ver
FROM u WHERE yy IS NOT NULL GROUP BY major ORDER BY pct_rev_2022plus DESC;

\echo === [B4] 수준 사다리: 대분류별 L5+ 능력단위 비율, 평균 수준
SELECT cp.major_category_name AS major, count(*) units,
       round(100.0*avg((cu.level >= 5)::int),1) AS pct_L5plus, round(avg(cu.level),2) AS mean_level,
       min(cu.level) AS minL, max(cu.level) AS maxL
FROM ncs.competency_units cu JOIN ncs.classification_paths cp ON cp.detail_category_full_code = cu.detail_category_full_code
GROUP BY 1 ORDER BY mean_level DESC;

\echo === [A2] 대분류별 KSA 고유 문장 비율 (중복 압축률)
SELECT cp.major_category_name AS major, count(*) AS kta, count(DISTINCT k.description) AS uniq,
       round(100.0*count(DISTINCT k.description)/count(*),1) AS pct_unique
FROM ncs.kta_items k
JOIN ncs.performance_criteria pc ON pc.id = k.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = pc.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
JOIN ncs.classification_paths cp ON cp.detail_category_full_code = cu.detail_category_full_code
GROUP BY 1 ORDER BY pct_unique;

\echo === [A4] 세분류 간 공유: 한 KSA 문장이 몇 개 직무(세분류)에 등장하는가 — 상위 15
SELECT k.description, count(DISTINCT cu.detail_category_full_code) AS n_jobs, count(*) AS n_occ
FROM ncs.kta_items k
JOIN ncs.performance_criteria pc ON pc.id = k.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = pc.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
GROUP BY 1 ORDER BY n_jobs DESC LIMIT 15;

\echo === [B2] 태도 어휘: 가장 흔한 태도 항목 상위 20 (등장 직무 수)
SELECT k.description, count(DISTINCT cu.detail_category_full_code) AS n_jobs
FROM ncs.kta_items k JOIN ncs.kta_types t ON t.code = k.kta_type_code
JOIN ncs.performance_criteria pc ON pc.id = k.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = pc.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
WHERE t.name LIKE '%태도%' GROUP BY 1 ORDER BY n_jobs DESC LIMIT 20;

\echo === [A1/B6] 수행준거 문장: 템플릿 준수율(…수 있다), 평균 길이 — 대분류별
SELECT cp.major_category_name AS major, count(*) AS pcs,
       round(100.0*avg((pc.description ~ '수 있다\.?\s*$')::int),1) AS pct_template,
       round(avg(length(pc.description)),1) AS mean_len
FROM ncs.performance_criteria pc
JOIN ncs.competency_unit_elements e ON e.code = pc.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
JOIN ncs.classification_paths cp ON cp.detail_category_full_code = cu.detail_category_full_code
GROUP BY 1 ORDER BY pct_template;

\echo === [B3] 디지털 어휘 침투: 대분류별 KSA 중 디지털/데이터/AI/SW/자동화 언급 비율
SELECT cp.major_category_name AS major,
       round(100.0*avg((k.description ~ '디지털|데이터|인공지능|AI|소프트웨어|자동화|프로그램|전산')::int),2) AS pct_digital
FROM ncs.kta_items k
JOIN ncs.performance_criteria pc ON pc.id = k.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = pc.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
JOIN ncs.classification_paths cp ON cp.detail_category_full_code = cu.detail_category_full_code
GROUP BY 1 ORDER BY pct_digital DESC;
