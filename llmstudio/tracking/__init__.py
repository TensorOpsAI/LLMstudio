import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import calendar
from sqlalchemy import extract, func
from sqlalchemy.orm import Session

from llmstudio.engine.providers import *
from llmstudio.tracking import crud, models, schemas
from llmstudio.tracking.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

TRACKING_HEALTH_ENDPOINT = "/health"
TRACKING_TITLE = "LLMstudio Tracking API"
TRACKING_DESCRIPTION = "The tracking API for LLM interactions"
TRACKING_VERSION = "0.0.1"
TRACKING_HOST = os.getenv("TRACKING_HOST", "localhost")
TRACKING_PORT = int(os.getenv("TRACKING_PORT", 8080))
TRACKING_URL = f"http://{TRACKING_HOST}:{TRACKING_PORT}"
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
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Dependency
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

    @app.post(f"{TRACKING_BASE_ENDPOINT}/projects", response_model=schemas.Project)
    def create_project(project: schemas.ProjectCreate, db: Session = Depends(get_db)):
        db_project = crud.get_project_by_name(db, name=project.name)
        if db_project:
            raise HTTPException(status_code=400, detail="Project already registered")
        return crud.create_project(db=db, project=project)

    @app.get(f"{TRACKING_BASE_ENDPOINT}/projects", response_model=list[schemas.Project])
    def read_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
        projects = crud.get_projects(db, skip=skip, limit=limit)
        return projects

    @app.get(
        f"{TRACKING_BASE_ENDPOINT}/projects/{{project_id}}",
        response_model=schemas.Project,
    )
    def read_project(project_id: int, db: Session = Depends(get_db)):
        db_project = crud.get_project(db, project_id=project_id)
        if db_project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return db_project

    @app.post(
        f"{TRACKING_BASE_ENDPOINT}/projects/{{project_id}}/sessions",
        response_model=schemas.Session,
    )
    def create_session(
        project_id: int, session: schemas.SessionCreate, db: Session = Depends(get_db)
    ):
        return crud.create_session(db=db, session=session, project_id=project_id)

    @app.get(f"{TRACKING_BASE_ENDPOINT}/sessions", response_model=list[schemas.Session])
    def read_sessions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
        sessions = crud.get_sessions(db, skip=skip, limit=limit)
        return sessions

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
        f"{TRACKING_BASE_ENDPOINT}/dashboard/metrics",
        response_model=schemas.DashboardMetrics,
    )
    def get_dashboard_metrics(
        db: Session = Depends(get_db), year: int = 2024, month: Optional[int] = None
    ):
        metrics_query = db.query(
            models.LogDefault.model,
            models.LogDefault.provider,
            func.count().label("total_requests"),
            extract("month", models.LogDefault.created_at).label("month"),
            extract("day", models.LogDefault.created_at).label("day"),
        ).filter(extract("year", models.LogDefault.created_at) == year)

        if month:
            metrics_query = metrics_query.filter(
                extract("month", models.LogDefault.created_at) == month
            )
            days_in_month = calendar.monthrange(year, month)[1]
        else:
            days_in_month = 12  # Months in a year

        metrics_query = metrics_query.group_by(
            "month", "day", models.LogDefault.model, models.LogDefault.provider
        ).all()

        # Get unique models and providers from the database
        unique_models = db.query(models.LogDefault.model).distinct().all()
        unique_providers = db.query(models.LogDefault.provider).distinct().all()

        # Flatten the lists of tuples to lists of strings
        unique_models = [model[0] for model in unique_models]
        unique_providers = [provider[0] for provider in unique_providers]

        total_cost_by_provider = (
            db.query(
                models.LogDefault.provider,
                func.sum(models.LogDefault.metrics["cost"]).label("cost"),
            )
            .group_by(models.LogDefault.provider)
            .all()
        )

        total_cost_by_model = (
            db.query(
                models.LogDefault.model,
                func.sum(models.LogDefault.metrics["cost"]).label("cost"),
            )
            .group_by(models.LogDefault.model)
            .all()
        )

        average_query = (
            db.query(
                models.LogDefault.model,
                func.avg(models.LogDefault.metrics["latency"]).label("average_latency"),
                func.avg(models.LogDefault.metrics["time_to_first_token"]).label(
                    "average_ttft"
                ),
                func.avg(models.LogDefault.metrics["inter_token_latency"]).label(
                    "average_itl"
                ),
                func.avg(models.LogDefault.metrics["tokens_per_second"]).label(
                    "average_tps"
                ),
            )
            .group_by(models.LogDefault.model)
            .all()
        )

        # Initialize the data structure with dates and zeros for all models and providers
        dashboard_data = {
            "request_by_provider": [
                {"date": index + 1, **{provider: 0 for provider in unique_providers}}
                for index in range(days_in_month)
            ],
            "request_by_model": [
                {"date": index + 1, **{model: 0 for model in unique_models}}
                for index in range(days_in_month)
            ],
            "total_cost_by_provider": [
                {"name": provider, "cost": cost}
                for provider, cost in total_cost_by_provider
            ],
            "total_cost_by_model": [
                {"name": model, "cost": cost} for model, cost in total_cost_by_model
            ],
            "average_latency": [
                {"name": model, "latency": average_latency}
                for model, average_latency, _, _, _ in average_query
            ],
            "average_ttft": [
                {"name": model, "ttft": average_ttft}
                for model, _, average_ttft, _, _ in average_query
            ],
            "average_itl": [
                {"name": model, "itl": average_itl}
                for model, _, _, average_itl, _ in average_query
            ],
            "average_tps": [
                {"name": model, "tps": average_tps}
                for model, _, _, _, average_tps in average_query
            ],
        }

        # Populate the data structure with actual counts from the query
        for metric in metrics_query:
            index = metric.month - 1 if not month else metric.day - 1
            dashboard_data["request_by_model"][index][
                metric.model
            ] = metric.total_requests
            dashboard_data["request_by_provider"][index][
                metric.provider
            ] = metric.total_requests

        return dashboard_data

    return app


def run_tracking_app():
    print(f"Running Tracking on http://{TRACKING_HOST}:{TRACKING_PORT}")
    try:
        tracking = create_tracking_app()
        uvicorn.run(
            tracking,
            host=TRACKING_HOST,
            port=TRACKING_PORT,
        )
    except Exception as e:
        print(f"Error running the Tracking app: {e}")
