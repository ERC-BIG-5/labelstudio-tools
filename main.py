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
    get_project,
)
from ls_helper.my_labelstudio_client.client import ls_client
from ls_helper.settings import SETTINGS
from ls_helper.tasks import strict_update_project_task_data
from tools.files import save_yaml
from tools.project_logging import get_logger

logger = get_logger(__file__)

app = typer.Typer(
    name="Labelstudio helper", pretty_exceptions_show_locals=False
)

app.add_typer(
    setup_app,
    name="setup",
    short_help="Commands related to initializing the project",
)
app.add_typer(
    project_app,
    name="project",
    short_help="Commands related to project setup and maintenance",
)
app.add_typer(
    backup_app,
    name="backup",
    short_help="Commands related to bulkbacking up projects and annotations",
)
app.add_typer(
    labeling_conf_app,
    name="labeling-conf",
    short_help="Commands related to building, validating and uploading project label configurations",
)
app.add_typer(
    task_app,
    name="task",
    short_help="Commands related to downloading, creating and patching project tasks",
)
app.add_typer(
    view_app, name="view", short_help="Commands related to project views"
)
app.add_typer(
    annotations_app,
    name="annotations",
    short_help="Commands related to downloading and analyzing annotations",
)
app.add_typer(
    pipeline_app,
    name="pipeline",
    short_help="Commands related to interaction with the Pipeline package",
)
app.add_typer(
    extras_app,
    name="extras",
    short_help="Some extra commands: [relational-values confusions]",
)

app.add_typer(
    aggregate_app,
    name="aggregate",
    short_help="Commands that run over multiple projects",
)
app.add_typer(plot_app,
              name="plot",
              short_help="Commands for plotting data")


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
    twitter = "twitter"
    youtube = "youtube"
    en = "en"
    es = "es"
    tw_es = {"platform": twitter, "language": es}
    yt_en4 = {"id": 50}
    _default = tw_es

    # setup

    # setup.add_projects()

    # this will work, since there is just one spanish twitter (so it's set to default)

    if False:
        from ls_helper.command.annotations import agreements

        agreements(
            **{"alias": "twitter-es-4"},
            accepted_ann_age=200,
            # variables=["coding-game"],
            exclude_variables=[
                "rel-value_text_conf_aesthetics",
                "rel-value_text_conf_cultural-identity",
                "rel-value_text_conf_social-cohesion",
                "rel-value_text_conf_good-life",
                "rel-value_text_conf_kinship",
                "rel-value_text_conf_livelihoods",
                "rel-value_text_conf_personal-identity",
                "rel-value_text_conf_reciprocity",
                "rel-value_text_conf_sense-of-agency",
                "rel-value_text_conf_sense-of-place",
                "rel-value_text_conf_social-relations",
                "rel-value_text_conf_social-responsibility",
                "rel-value_text_conf_spirituality",
                "rel-value_text_conf_stewardship-principle",
                "rel-value_text_conf_well-being",
                "rel-value_visual_conf_aesthetics",
                "rel-value_visual_conf_cultural-identity",
                "rel-value_visual_conf_social-cohesion",
                "rel-value_visual_conf_good-life",
                "rel-value_visual_conf_kinship",
                "rel-value_visual_conf_livelihoods",
                "rel-value_visual_conf_personal-identity",
                "rel-value_visual_conf_reciprocity",
                "rel-value_visual_conf_sense-of-agency",
                "rel-value_visual_conf_sense-of-place",
                "rel-value_visual_conf_social-relations",
                "rel-value_visual_conf_social-responsibility",
                "rel-value_visual_conf_spirituality",
                "rel-value_visual_conf_stewardship-principle",
                "rel-value_visual_conf_well-being",
            ],
        )

    # reconfigure twitter-en/es pv.5

    # redo, label configs for # p 5

    # for i in [54]:
    #     # labeling_conf.build(i)
    #     # labeling_conf.update_labeling_config(i)
    #     project_setup.generate_variable_extensions_template(
    #         id=i, overwrite_if_exists=False
    #     )

    # labeling_conf.build_ls_labeling_interface(id=53)

    # print(get_variables_info(53, from_built=True))

    # labeling_conf.build_extension_index(False,[51,50])
    """project_setup.create_project(
        title="Twitter - ES - protocol.v5",
        alias="twitter-es-5",
        platform="twitter",
        language="en",
    )"""
    # project_setup.generate_variable_extensions_template(53)
    # project_setup.generate_variable_extensions_template(54)

    # create_conflict_view("nature_any",**{"alias": "twitter-es-4"})
    # get_tasks(**{"alias": "twitter-es-4"})
    # add_conflicts_to_tasks(**{"alias": "twitter-es-4"})

    """
    # for creating/testing version 5 of the protocol
    from ls_helper.command import labeling_conf

    p, valid = labeling_conf.build_ls_labeling_interface(**{"id": 53})

    if valid:
        labeling_conf.update_labeling_config(**{"id": 53})

    yaml.dump(get_variables_info(id=53), Path("53.yaml").open("w", encoding="utf-8"))
    """
    # get_confusions(id=51)
    # add_conflicts_to_tasks(id=51)
    # get_confusions(id=51)
    # update_coding_game(id=51)
    # from ls_helper.command import pipeline
    # pipeline.reformat_for_datapipelines(id=43)
    # pipeline.reformat_for_datapipelines(alias="twitter-es-4")

    # overview()

    # project_setup.generate_variable_extensions_template(43)
    # from ls_helper.command import annotations

    """ extras, food, health
    project_setup.download_project_data(50)
    # project_setup.generate_variable_extensions_template(id=50)
    for i in [43, 50, 51]:  # tw-en,yt-en, tw-es,
        annotations.clean_results(i, variables={"extras"})
    """
    # for i in [43,51]:  # tw-en,yt-en, tw-es,
    #     annotations.clean_results(i, variables={"nature_any","nature_text","nature_visual_any"})

    # task.get_tasks(id=54)
    # annotations(**{"id": 54, "accepted_ann_age": 0})

    # po = get_project(id=51)
    # res = po.get_annotations_results()
    # # df = res.flatten_annotation_results()
    # raw_df, _ = res.get_annotation_df(
    #     debug_tasks=[33000], debug_task_limit=10, test_rebuild=True
    # )
    # df = res.flatten_annotation_results()
    # pass

    # aggregate.check_projects([51, 53, 54])
    # get_project(54).refresh_views()

    from ls_helper.command import view, annotations

    view.download_project_views(id=53)
    # annotations.agreements(id=54)
    # task.get_tasks(id=53)
    # view.update_coding_game(id=53)
    annotations.add_conflicts_to_tasks(id=54)
