from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from main import update_coding_game

app = FastAPI(root_path="/DATA")
app.mount("/static", StaticFiles(directory="data/server_static"), name="static")


@app.get("/")
def index():
    return FileResponse("data/server_static/index.html")


@app.get("/update-coding-game")
def _update_coding_game(platform: str, language: str):
    res = update_coding_game(platform, language)
    if res:
        project_id, view_id = res
        return project_id, view_id
    else:
        raise HTTPException(status_code=404)
