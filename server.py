from pathlib import Path
from typing import Annotated, Optional

from ls_helper.command.extra import get_confusions

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
    from starlette.responses import FileResponse
    from starlette.staticfiles import StaticFiles
except ImportError:
    print("You need to install the optional dependency [server]")
from main import (
    get_all_variable_names,
    update_coding_game,
)
from ls_helper.command.view import create_conflict_view
from ls_helper.command.annotations import agreements

# app = FastAPI()
app = FastAPI(root_path="/")

app.mount(
    "/static", StaticFiles(directory="data/server_static"), name="static"
)
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
def _annotations_results(
    platform: str, language: str, annotation_age: int = 2
):
    # todo redo
    """

    :param platform:
    :param language:
    :param annotation_age:
    :return:
    """

    """
    file_path, annot_orig = annotations(platform, language, annotation_age)
    fn = f"{file_path.stem}_{annot_orig}.csv"
    return FileResponse(
        path=file_path,
        filename=fn,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fn}"}
    )"""


@app.get("/agreements")
def _agreements(platform: str, language: str, annotation_age: int = 2):
    file_paths_dict = agreements(platform, language, annotation_age)
    for k, p in file_paths_dict.items():
        file_paths_dict[k] = Path("data") / p.relative_to("data/ls_data")
    return file_paths_dict


@app.get("/get-all-variable-names")
def _get_all_variable_names(
    project_access: Annotated[ProjectAccessQuery, Query()],
):
    return get_all_variable_names(
        project_access.id,
        project_access.aliat,
        project_access.platform,
        project_access.language,
    )


@app.get("/create-conflict-view")
def _create_conflict_view(
    variable: Annotated[str, Query(description="conflict 'choice' variable")],
    id: Optional[int] = None,
    alias: Optional[str] = None,
    platform: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    return create_conflict_view(variable, id, alias, platform, language)


@app.get("/confusion-analysis")
def _confusion_analysis(
    alias: Annotated[str, Query(description="alias of the project")],
):
    try:
        file_path = get_confusions(alias=alias, accepted_ann_age=300)
        filename = f"confusion-{alias}.csv"
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )

    except Exception:
        return "Sorry, something went wrong. Contact Ramin", 500
