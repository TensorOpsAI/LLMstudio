broker_url = "pyamqp://guest:guest@rabbitmq//"
result_backend = "redis://localhost:6379/0"
task_serializer = "json"
accept_content = ["json"]
result_serializer = "json"
timezone = "UTC"
enable_utc = True
imports = ("api.worker.tasks",)
