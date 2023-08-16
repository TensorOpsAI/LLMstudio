from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from endpoints.chat.openai import router as chat_openai_router
from endpoints.test.openai import router as test_openai_router
from endpoints.chat.vertexai import router as chat_vertexai_router
from endpoints.test.vertexai import router as test_vertexai_router
from endpoints.export import router as export_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_openai_router, prefix="/api/chat")
app.include_router(test_openai_router, prefix="/api/test")
app.include_router(chat_vertexai_router, prefix="/api/chat")
app.include_router(test_vertexai_router, prefix="/api/test")
app.include_router(export_router, prefix="/api")
