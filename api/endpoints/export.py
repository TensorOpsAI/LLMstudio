import json
from typing import Any, Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.post("/export", response_class=StreamingResponse)
async def export(request: Request):
    """
    Export data in CSV format.

    This API endpoint accepts JSON data via POST, converts it into CSV format,
    and returns it as a streaming response. This allows the downloading of the data
    as a CSV file named 'parameters.csv'.

    Args:
    request (Request): The incoming request, which contains the JSON data
                         to be converted to CSV.

    Returns:
    StreamingResponse: A FastAPI response class that streams the CSV content
                         back to the client, prompting a download of a file named 'myplot.csv'.
    """
    data = await request.json()
    csv_content = ""

    if len(data) > 0:
        csv_content += ";".join(data[0].keys()) + "\n"
        for execution in data:
            csv_content += ";".join([json.dumps(value) for value in execution.values()]) + "\n"

    headers = {"Content-Disposition": "attachment; filename=parameters.csv"}
    return StreamingResponse(iter([csv_content]), media_type="text/csv", headers=headers)
