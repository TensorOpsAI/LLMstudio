from app import app
from flask import Response, send_file, request
import json


@app.route("/export", methods=["POST"])
def export():
    csv = ""

    if len(request.json) > 0:
        csv += ";".join(request.json[0].keys()) + "\n"
        for execution in request.json:
            for value in execution.values():
                print(type(value))
            csv += ";".join([json.dumps(value) for value in execution.values()]) + "\n"
            print(csv)

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=myplot.csv"},
    )
