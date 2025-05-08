from typing import List

from fastapi import APIRouter, Depends
from llmstudio_tracker.database import get_db
from llmstudio_tracker.db.crud import session as session_crud
from llmstudio_tracker.db.schemas.session import (
    SessionDefaultCreate,
    SessionDefaultResponse,
)
from sqlalchemy.orm import Session


class SessionsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router
        self.define_routes()

    def define_routes(self):
        # Add session
        self.router.post("/session", response_model=SessionDefaultResponse)(
            self.add_session
        )

        # Read session
        self.router.get(
            "/session/{session_id}", response_model=List[SessionDefaultResponse]
        )(self.get_session)

        self.router.patch(
            "/session/{message_id}", response_model=SessionDefaultResponse
        )(self.update_session)

    async def add_session(
        self, session: SessionDefaultCreate, db: Session = Depends(get_db)
    ):
        return session.upsert_session(db=db, session=session)

    async def update_session(
        self, message_id: int, extras: dict, db: Session = Depends(get_db)
    ):
        sessions = session_crud.update_session(db, message_id=message_id, extras=extras)
        return sessions

    async def get_session(self, session_id: str, db: Session = Depends(get_db)):
        sessions = session_crud.get_session_by_session_id(db, session_id=session_id)
        return sessions
