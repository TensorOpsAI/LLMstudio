from typing import List

from fastapi import APIRouter, Depends
from llmstudio_tracker.database import get_db
from llmstudio_tracker.db.crud import logs as logs_crud
from llmstudio_tracker.db.schemas.logs import LogDefaultCreate, LogDefaultResponse
from sqlalchemy.orm import Session


class LogsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router

        # Define routes
        self.define_routes()

    def define_routes(self):
        # Add log
        self.router.post(
            "/logs",
            response_model=LogDefaultResponse,
        )(self.add_log)

        # Read logs
        self.router.get("/logs", response_model=List[LogDefaultResponse])(
            self.read_logs
        )

        # Read logs by session
        self.router.get("/logs/{session_id}", response_model=List[LogDefaultResponse])(
            self.read_logs_by_session
        )

    async def add_log(self, log: LogDefaultCreate, db: Session = Depends(get_db)):
        return logs_crud.add_log(db=db, log=log)

    async def read_logs(
        self, skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)
    ):
        logs = logs_crud.get_logs(db, skip=skip, limit=limit)
        return logs

    async def read_logs_by_session(
        self,
        session_id: str,
        skip: int = 0,
        limit: int = 1000,
        db: Session = Depends(get_db),
    ):
        logs = logs_crud.get_logs_by_session(
            db, session_id=session_id, skip=skip, limit=limit
        )
        return logs
