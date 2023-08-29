from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from endpoints.chat.openai import router as chat_openai_router
from endpoints.test.openai import router as test_openai_router
from endpoints.chat.vertexai import router as chat_vertexai_router
from endpoints.test.vertexai import router as test_vertexai_router
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


@app.websocket("/ws/{channel_name}")
async def websocket_endpoint(websocket: WebSocket, channel_name: str):
    await websocket.accept()
    pubsub.subscribe(channel_name)

    for message in pubsub.listen():
        if message["type"] == "message":
            await websocket.send_text(message["data"].decode("utf-8"))


app.include_router(chat_openai_router, prefix="/api/chat")
app.include_router(test_openai_router, prefix="/api/test")
app.include_router(chat_vertexai_router, prefix="/api/chat")
app.include_router(test_vertexai_router, prefix="/api/test")
app.include_router(export_router, prefix="/api")
