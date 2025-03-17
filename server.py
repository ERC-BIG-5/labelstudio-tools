from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from main import update_coding_game, annotations_results, agreements

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
    file_path = agreements(platform, language, annotation_age)
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={file_path.name}"}
    )