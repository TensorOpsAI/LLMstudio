from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from endpoints.chat.openai import router as chat_openai_router
from endpoints.test.openai import router as test_openai_router
from endpoints.chat.vertexai import router as chat_vertexai_router
from endpoints.test.vertexai import router as test_vertexai_router
from endpoints.chat.bedrock import router as chat_bedrock_router
from endpoints.test.bedrock import router as test_bedrock_router
from endpoints.export import router as export_router

from api.worker.config import pubsub

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/logs")
def read_logs():
    return FileResponse("/api/logs/execution_logs.jsonl", media_type="application/json")


app.include_router(chat_openai_router, prefix="/api/chat")
app.include_router(test_openai_router, prefix="/api/test")
app.include_router(chat_vertexai_router, prefix="/api/chat")
app.include_router(test_vertexai_router, prefix="/api/test")
app.include_router(chat_bedrock_router, prefix="/api/chat")
app.include_router(test_bedrock_router, prefix="/api/test")
app.include_router(export_router, prefix="/api")
