CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    cmetadata JSONB
);

CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL,
    embedding VECTOR(384) NOT NULL,
    document TEXT,
    cmetadata JSONB,
    custom_id VARCHAR,
    CONSTRAINT fk_collection
        FOREIGN KEY (collection_id)
        REFERENCES langchain_pg_collection(uuid)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_idx
    ON langchain_pg_embedding
    USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);