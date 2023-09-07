import json


def append_log(data):
    path = "/api/logs/execution_logs.jsonl"

    with open(path, "a") as file:
        file.write(json.dumps(data) + "\n")
