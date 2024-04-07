from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from llmstudio.tracking.database import engine, get_db
from llmstudio.tracking.session import crud, models, schemas

models.Base.metadata.create_all(bind=engine)


class SessionsRoutes:
    def __init__(self, router: APIRouter):
        self.router = router
        self.define_routes()

    def define_routes(self):
        # Add session
        self.router.post(
            "/session",
            response_model=schemas.SessionDefault,
        )(self.add_session)

        # Read session
        self.router.get("/session/{session_id}", response_model=schemas.SessionDefault)(
            self.get_session
        )

    async def add_session(
        self, session: schemas.SessionDefaultCreate, db: Session = Depends(get_db)
    ):
        return crud.upsert_session(db=db, session=session)

    async def get_session(self, session_id: str, db: Session = Depends(get_db)):
        logs = crud.get_session_by_id(db, session_id=session_id)
        return logs
