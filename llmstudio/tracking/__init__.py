import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import extract, func
from sqlalchemy.orm import Session

from llmstudio.config import TRACKING_HOST, TRACKING_PORT
from llmstudio.engine.providers import *
from llmstudio.tracking import crud, models, schemas
from llmstudio.tracking.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

TRACKING_HEALTH_ENDPOINT = "/health"
TRACKING_TITLE = "LLMstudio Tracking API"
TRACKING_DESCRIPTION = "The tracking API for LLM interactions"
TRACKING_VERSION = "0.0.1"
TRACKING_BASE_ENDPOINT = "/api/tracking"


## Tracking
def create_tracking_app() -> FastAPI:
    app = FastAPI(
        title=TRACKING_TITLE,
        description=TRACKING_DESCRIPTION,
        version=TRACKING_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    @app.get(TRACKING_HEALTH_ENDPOINT)
    def health_check():
        """Health check endpoint to ensure the API is running."""
        return {"status": "healthy", "message": "Tracking is up and running"}

    @app.post(
        f"{TRACKING_BASE_ENDPOINT}/logs",
        response_model=schemas.LogDefault,
    )
    def add_log(log: schemas.LogDefaultCreate, db: Session = Depends(get_db)):
        return crud.add_log(db=db, log=log)

    @app.get(f"{TRACKING_BASE_ENDPOINT}/logs", response_model=list[schemas.LogDefault])
    def read_logs(skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)):
        logs = crud.get_logs(db, skip=skip, limit=limit)
        return logs

    @app.get(
        f"{TRACKING_BASE_ENDPOINT}/logs_by_session",
        response_model=list[schemas.LogDefault],
    )
    def read_logs_by_session(
        session_id: str, skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)
    ):
        logs = crud.get_logs_by_session(
            db, session_id=session_id, skip=skip, limit=limit
        )
        return logs

    @app.on_event("startup")
    async def startup_event():
        print(f"Running LLMstudio Tracking on http://{TRACKING_HOST}:{TRACKING_PORT} ")

    return app


def run_tracking_app():
    try:
        tracking = create_tracking_app()
        uvicorn.run(
            tracking,
            host=TRACKING_HOST,
            port=TRACKING_PORT,
            log_level="warning",
        )
    except Exception as e:
        print(f"Error running LLMstudio Tracking: {e}")


if __name__ == "__main__":
    run_tracking_app()
