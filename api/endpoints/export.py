from fastapi import APIRouter, Request
from typing import List, Any, Dict
from fastapi.responses import StreamingResponse
import json

router = APIRouter()


@router.post("/export", response_class=StreamingResponse)
async def export(request: Request):
    data = await request.json()
    csv_content = ""

    if len(data) > 0:
        csv_content += ";".join(data[0].keys()) + "\n"
        for execution in data:
            csv_content += (
                ";".join([json.dumps(value) for value in execution.values()]) + "\n"
            )

    headers = {"Content-Disposition": "attachment; filename=myplot.csv"}
    return StreamingResponse(
        iter([csv_content]), media_type="text/csv", headers=headers
    )
