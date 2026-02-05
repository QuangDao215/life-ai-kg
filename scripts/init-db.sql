-- PostgreSQL initialization script
-- This runs automatically when the container is first created

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- Create a read-only user for analytics (optional)
-- CREATE USER lifeai_readonly WITH PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE lifeai_kg TO lifeai_readonly;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully';
END $$;
