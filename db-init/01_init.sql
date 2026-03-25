-- ============================================================
--  db-init/01_init.sql
--  Crea los esquemas y tablas para el proyecto MLOps Penguins
-- ============================================================

-- ── Tabla de datos crudos (raw) ────────────────────────────
CREATE TABLE IF NOT EXISTS penguins_raw (
    id                SERIAL PRIMARY KEY,
    species           VARCHAR(50),
    island            VARCHAR(50),
    bill_length_mm    FLOAT,
    bill_depth_mm     FLOAT,
    flipper_length_mm FLOAT,
    body_mass_g       FLOAT,
    sex               VARCHAR(10),
    year              INT,
    loaded_at         TIMESTAMP DEFAULT NOW()
);

-- ── Tabla de datos procesados (features para entrenamiento) ─
CREATE TABLE IF NOT EXISTS penguins_processed (
    id                    SERIAL PRIMARY KEY,
    bill_length_mm        FLOAT   NOT NULL,
    bill_depth_mm         FLOAT   NOT NULL,
    flipper_length_mm     FLOAT   NOT NULL,
    body_mass_g           FLOAT   NOT NULL,
    island_Dream          INT     NOT NULL DEFAULT 0,
    island_Torgersen      INT     NOT NULL DEFAULT 0,
    sex_male              INT     NOT NULL DEFAULT 0,
    species               INT     NOT NULL,   -- 0=Adelie 1=Chinstrap 2=Gentoo
    species_label         VARCHAR(20) NOT NULL,
    split                 VARCHAR(10) NOT NULL DEFAULT 'train', -- train / test
    processed_at          TIMESTAMP DEFAULT NOW()
);

-- ── Tabla de predicciones (log de inferencias) ──────────────
CREATE TABLE IF NOT EXISTS predictions_log (
    id                    SERIAL PRIMARY KEY,
    run_id                VARCHAR(100),
    model_version         VARCHAR(50),
    bill_length_mm        FLOAT,
    bill_depth_mm         FLOAT,
    flipper_length_mm     FLOAT,
    body_mass_g           FLOAT,
    island                VARCHAR(50),
    sex                   VARCHAR(10),
    predicted_species     VARCHAR(20),
    predicted_class       INT,
    confidence            FLOAT,
    requested_at          TIMESTAMP DEFAULT NOW()
);

-- índices útiles
CREATE INDEX IF NOT EXISTS idx_raw_species   ON penguins_raw(species);
CREATE INDEX IF NOT EXISTS idx_proc_split    ON penguins_processed(split);
CREATE INDEX IF NOT EXISTS idx_pred_run_id   ON predictions_log(run_id);
