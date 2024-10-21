from typing import List

from fastapi import APIRouter, Depends
from llmstudio_tracker.database import engine, get_db
from llmstudio_tracker.logs import crud, models, schemas
from sqlalchemy.orm import Session

models.Base.metadata.create_all(bind=engine)


class LogsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router

        # Define routes
        self.define_routes()

    def define_routes(self):
        # Add log
        self.router.post(
            "/logs",
            response_model=schemas.LogDefault,
        )(self.add_log)

        # Read logs
        self.router.get("/logs", response_model=List[schemas.LogDefault])(
            self.read_logs
        )

        # Read logs by session
        self.router.get("/logs/{session_id}", response_model=List[schemas.LogDefault])(
            self.read_logs_by_session
        )

    async def add_log(
        self, log: schemas.LogDefaultCreate, db: Session = Depends(get_db)
    ):
        return crud.add_log(db=db, log=log)

    async def read_logs(
        self, skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)
    ):
        logs = crud.get_logs(db, skip=skip, limit=limit)
        return logs

    async def read_logs_by_session(
        self,
        session_id: str,
        skip: int = 0,
        limit: int = 1000,
        db: Session = Depends(get_db),
    ):
        logs = crud.get_logs_by_session(
            db, session_id=session_id, skip=skip, limit=limit
        )
        return logs
