from fastapi import FastAPI
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from main import update_coding_game

app = FastAPI()
update_coding_game
app.mount("/static", StaticFiles(directory="data/server_static"), name="static")


@app.get("/")
def index():
    return FileResponse("data/server_static/index.html")


@app.get("/update-coding-game")
def _update_coding_game(platform: str, language: str):
    project_id, view_id = update_coding_game(platform, language)
    return project_id, view_id
