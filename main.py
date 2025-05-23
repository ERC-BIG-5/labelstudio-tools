import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from ls_helper.command.aggregate import aggregate_app
from ls_helper.command.annotations import annotations_app
from ls_helper.command.backup import backup_app
from ls_helper.command.extra import extras_app
from ls_helper.command.labeling_conf import labeling_conf_app
from ls_helper.command.pipeline import pipeline_app
from ls_helper.command.plot import plot_app
from ls_helper.command.project_setup import project_app
from ls_helper.command.setup import (
    setup_app,
)
from ls_helper.command.task import task_add_predictions, task_app
from ls_helper.command.view import view_app
from ls_helper.config_helper import parse_label_config_xml
from ls_helper.models.interface_models import IChoices, InterfaceData
from ls_helper.models.main_models import (
    get_project, )
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.settings import SETTINGS
from ls_helper.tasks import strict_update_project_task_data
from tools.files import save_yaml
from tools.project_logging import get_logger

logger = get_logger(__file__)

app = typer.Typer(
    name="Labelstudio helper", pretty_exceptions_show_locals=False
)

sub_apps = [
    (setup_app, "setup", "Commands related to initializing the project"),
    (project_app, "project", "Commands related to project setup and maintenance"),
    (backup_app, "backup", "Commands related to backing up projects and annotations in bulk"),
    (labeling_conf_app, "labeling-conf",
     "Commands related to building, validating and uploading project label configurations"),
    (task_app, "task", "Commands related to downloading, creating and patching project tasks"),
    (view_app, "view", "Commands related to project views"),
    (annotations_app, "annotations", "Commands related to downloading and analyzing annotations"),
    (pipeline_app, "pipeline", "Commands related to interaction with the Pipeline package"),
    (extras_app, "extras", "Some extra commands: [relational-values confusions]"),
    (aggregate_app, "aggregate", "Commands that run over multiple projects"),
    (plot_app, "plot", "Commands for plotting data")
]

for sub_app, name, help_text in sub_apps:
    app.add_typer(sub_app, name=name, help=help_text)


@app.command(
    name="strict task update",
    short_help="[ls maint] Update tasks. Files must be matching lists of {id: , data:}",
)
# todo: more testing
def strict_update_project_tasks(
        new_data_file: Path, existing_data_file: Optional[Path] = None
):
    raise NotImplementedError("client.patch_task parameters changed")
    new_data_list = json.loads(new_data_file.read_text(encoding="utf-8"))
    if existing_data_file:
        existing_data_list = json.loads(
            existing_data_file.read_text(encoding="utf-8")
        )
        assert len(new_data_list) == len(existing_data_list)

        for idx, t in tqdm(enumerate(new_data_list)):
            t_id = t["id"]
            ex_t = existing_data_list[idx]
            assert t_id == ex_t["id"]
            strict_update_project_task_data(t_id, t, ex_t)

        print(f"{len(new_data_list)} tasks updated")
        return

    for t in tqdm(new_data_list):
        ls_client().patch_task(t["id"], t["data"])

    print(f"{len(new_data_list)} tasks updated")


def get_all_variable_names(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
):
    po = get_project(id, alias, platform, language)
    # todo redo and test...
    struct = po.raw_interface_struct
    return list(struct.ordered_fields_map.keys())


def get_variables_info(
        id: Annotated[Optional[int], typer.Option()] = None,
        alias: Annotated[Optional[str], typer.Option("-a")] = None,
        platform: Annotated[Optional[str], typer.Option()] = None,
        language: Annotated[Optional[str], typer.Option()] = None,
        from_built: Annotated[
            bool,
            typer.Option(False, help="Use the built instead of the project-data"),
        ] = False,
):
    po = get_project(id, alias, platform, language)

    if from_built:
        config: InterfaceData = parse_label_config_xml(
            po.path_for(
                SETTINGS.built_labeling_configs, ext=".xml"
            ).read_text()
        )
    else:
        config = po.raw_interface_struct

    interface_data = [
        {
            "name": k,
            "required": v.required,
            "choice_type": str(v.choice)
            if isinstance(v, IChoices)
            else "text",
            "choices": v.raw_options_list() if isinstance(v, IChoices) else "",
        }
        for k, v in config.ordered_fields_map.items()
    ]

    # yaml.dump(interface_data, po.path_for(SETTINGS.temp_file_path, ext=".yaml").open("w"), indent=True, default_flow_style=False, encoding="utf-8")
    save_yaml(
        po.path_for(SETTINGS.temp_file_path, ext=".yaml"), interface_data
    )
    return interface_data


def add_prediction_test():
    resp = task_add_predictions(
        33030,
        {
            "model_version": "one",
            "score": 0.5,
            # "type": "choices",
            "result": [
                {
                    # "id": "result1",
                    "type": "choices",
                    "to_name": "title",
                    "from_name": "nature_any",
                    "value": {"choices": ["Yes"]},
                },
                {
                    # "id": "result1",
                    "type": "choices",
                    "to_name": "title",
                    "from_name": "nature_visual",
                    "value": {"choices": ["Yes"]},
                },
            ],
        },
    )
    print(json.dumps(resp.json(), indent=2))


@app.command(name="overview", short_help="Overview of all commands")
def overview():
    def _print_commands(current_app, prefix="", indent=0):
        """Recursively print commands and subcommands."""
        indent_str = "  " * indent

        # Print commands at this level
        for cmd in current_app.registered_commands:
            name = cmd.name or cmd.callback.__name__
            help_text = cmd.short_help or ""
            print(f"{indent_str}• '{name}' - {help_text}")

        # Find and process all subapps
        for group in current_app.registered_groups:
            subapp_name = group.name
            subapp = group.typer_instance
            print(f"{indent_str}▼ {prefix}{subapp_name}")
            _print_commands(subapp, f"{prefix}{subapp_name} ", indent + 1)

    # Start the recursive printing from the main app
    _print_commands(app)


if __name__ == "__main__":
    try:
        from ls_helper.local_main import main
    except ImportError:
        print("You wanna create 'ls_helper.local_main' with a main function... its in gitignore and we keep main clean")
