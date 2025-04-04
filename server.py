from pathlib import Path
from typing import Optional, Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from starlette.requests import Request
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ls_helper.models import MyProject
from main import update_coding_game, annotations, agreements, create_conflict_view, get_all_variable_names

# app = FastAPI()
app = FastAPI(root_path="/DATA")

app.mount("/static", StaticFiles(directory="data/server_static"), name="static")
app.mount("/data", StaticFiles(directory="data/ls_data"), name="data")


class ProjectAccessQuery(BaseModel):
    id: int = None
    aliat: str = None
    platform: str = None
    language: str = None


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
    file_path, annot_orig = annotations(platform, language, annotation_age)
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


@app.get("/get-all-variable-names")
def _get_all_variable_names(project_access: Annotated[ProjectAccessQuery, Query()]):
    return get_all_variable_names(project_access.id,
                                  project_access.aliat,
                                  project_access.platform,
                                  project_access.language)


@app.get("/create-conflict-view")
def _create_conflict_view(variable: Annotated[str, Query(description="conflict 'choice' variable")],
                          id: Optional[int] = None,
                          alias: Optional[str] = None, platform: Optional[str] = None,
                          language: Optional[str] = None) -> str:
    return create_conflict_view(variable, id, alias, platform, language)
