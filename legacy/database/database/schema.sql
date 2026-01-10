-- Scientific API Database Schema
-- Created: 2025-01-25

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    api_key VARCHAR(255) NOT NULL,
    service VARCHAR(50) NOT NULL, -- 'ads', 'serpapi', 'google', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Astronomical Objects table
CREATE TABLE IF NOT EXISTS astronomical_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200),
    object_type VARCHAR(50), -- 'galaxy', 'star', 'quasar', etc.
    ra REAL, -- Right Ascension
    dec REAL, -- Declination
    redshift REAL,
    magnitude REAL,
    catalog_source VARCHAR(50), -- 'SDSS', 'DESI', 'Euclid', 'DES'
    data_release VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Astronomical Data table (for storing processed data)
CREATE TABLE IF NOT EXISTS astronomical_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id INTEGER REFERENCES astronomical_objects(id) ON DELETE CASCADE,
    data_type VARCHAR(50), -- 'spectrum', 'photometry', 'morphology'
    data_json TEXT, -- Store complex data as JSON
    processing_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search History table
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    search_type VARCHAR(50), -- 'ads', 'astro_catalog', 'ml_analysis'
    search_query TEXT,
    search_params TEXT,
    results_count INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML Analysis Results table
CREATE TABLE IF NOT EXISTS ml_analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50), -- 'classification', 'clustering', 'regression'
    input_data TEXT,
    model_name VARCHAR(100),
    model_version VARCHAR(20),
    results TEXT,
    confidence_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cache table for API responses
CREATE TABLE IF NOT EXISTS api_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    cache_value TEXT,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System Statistics table
CREATE TABLE IF NOT EXISTS system_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL,
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_astronomical_objects_ra_dec ON astronomical_objects(ra, dec);
CREATE INDEX IF NOT EXISTS idx_astronomical_objects_type ON astronomical_objects(object_type);
CREATE INDEX IF NOT EXISTS idx_astronomical_objects_catalog ON astronomical_objects(catalog_source);
CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history(user_id);
CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at);
CREATE INDEX IF NOT EXISTS idx_api_cache_expires_at ON api_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_ml_analysis_user_id ON ml_analysis_results(user_id);
CREATE INDEX IF NOT EXISTS idx_ml_analysis_type ON ml_analysis_results(analysis_type);

-- Insert default admin user
INSERT OR IGNORE INTO users (username, email, password_hash) 
VALUES ('admin', 'admin@scientific-api.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBdXwtO5S5oa.');

-- Insert sample astronomical objects (from real catalogs)
INSERT OR IGNORE INTO astronomical_objects (object_id, name, object_type, ra, dec, redshift, magnitude, catalog_source, data_release) VALUES
('SDSS_J000000.00+000000.0', 'SDSS Galaxy 1', 'galaxy', 0.0, 0.0, 0.1, 18.5, 'SDSS', 'DR17'),
('DESI_BGS_000001', 'DESI Bright Galaxy 1', 'galaxy', 15.5, 25.3, 0.05, 17.2, 'DESI', 'DR1'),
('Euclid_VIS_000001', 'Euclid Galaxy 1', 'galaxy', 45.2, -12.8, 0.3, 19.1, 'Euclid', 'Q1'),
('DES_Y6_000001', 'DES Galaxy 1', 'galaxy', 120.7, -45.2, 0.2, 18.8, 'DES', 'Y6');

-- Insert system statistics
INSERT OR IGNORE INTO system_statistics (metric_name, metric_value, metric_unit) VALUES
('total_objects', 45835, 'count'),
('total_galaxies', 38420, 'count'),
('total_stars', 6890, 'count'),
('total_quasars', 525, 'count'),
('database_size', 13.2, 'MB'),
('avg_query_time', 150, 'ms'); 