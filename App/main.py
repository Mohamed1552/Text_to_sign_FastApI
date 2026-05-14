from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.routes.text_to_sign import text_router
from api.routes.Speech_to_text import speech_router

app = FastAPI()

app.include_router(text_router)

app.include_router(speech_router)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)