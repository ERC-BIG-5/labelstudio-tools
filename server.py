from pathlib import Path

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from main import update_coding_game, annotations_results, agreements

# app = FastAPI()
app = FastAPI(root_path="/DATA")

app.mount("/static", StaticFiles(directory="data/server_static"), name="static")
app.mount("/data", StaticFiles(directory="data/ls_data"), name="data")


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


@app.get("/annotations-results")
def _annotations_results(platform: str, language: str, annotation_age: int = 2):
    file_path, annot_orig = annotations_results(platform, language, annotation_age)
    fn = f"{file_path.stem}_{annot_orig}.csv"
    return FileResponse(
        path=file_path,
        filename=fn,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fn}"}
    )


@app.get("/agreements")
def _agreements(platform: str, language: str, annotation_age: int = 2):
    file_paths_dict = agreements(platform, language, annotation_age)
    for k, p in file_paths_dict.items():
        file_paths_dict[k] = Path("data") / p.relative_to("data/ls_data")
    return file_paths_dict
