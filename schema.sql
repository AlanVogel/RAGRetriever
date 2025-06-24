CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL,
    embedding VECTOR(384) NOT NULL,
    document TEXT,
    cmetadata JSONB,
    CONSTRAINT fk_collection
        FOREIGN KEY (collection_id)
        REFERENCES langchain_pg_collection(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_idx
    ON langchain_pg_embedding
    USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);