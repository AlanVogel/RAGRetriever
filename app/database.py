from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def initialize_database(connection_string):
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Session = sessionmaker(bind=engine)
    return Session
