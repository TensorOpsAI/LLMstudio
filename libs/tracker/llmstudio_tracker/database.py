from llmstudio_tracker.config import DB_TYPE, TRACKING_URI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_tracking_engine(uri: str):
    if DB_TYPE == "sqlite":
        return create_engine(uri, connect_args={"check_same_thread": False})
    return create_engine(uri)


engine = create_tracking_engine(TRACKING_URI)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
