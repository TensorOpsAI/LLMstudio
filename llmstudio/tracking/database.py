import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(os.path.join(os.getcwd(), ".env"))

SQLALCHEMY_DATABASE_URL = os.environ.get(
    "BACKEND_TRACKING_URI", "sqlite:///./llmstudio_mgmt.db"
)
# It works with any sqlalchemy table. host your postgres by changing it to something like
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
