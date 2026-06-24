from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.routes.text_to_sign import text_router
from api.routes.Speech_to_text import speech_router
from api.routes.video import video_router
from api.routes.sign_to_text import router as sign_to_text_router
from api.routes.full_sentence_video import router as sentence_video_router

app = FastAPI()

app.include_router(text_router)

app.include_router(speech_router)

app.include_router(video_router)

app.include_router(sign_to_text_router)

app.include_router(sentence_video_router)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)