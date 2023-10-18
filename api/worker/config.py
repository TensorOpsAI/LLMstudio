import redis
from celery import Celery

celery_app = Celery(
    "tasks",
    broker="pyamqp://guest@rabbitmq//",
    backend="rpc://",
)

celery_app.config_from_object("api.config.celery")
celery_app.autodiscover_tasks(["api.worker.tasks"])

redis_conn = redis.StrictRedis(host="localhost", port=6379, db=0)
pubsub = redis_conn.pubsub()
