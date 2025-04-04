import xml.etree.ElementTree as ET
from pathlib import Path

from deepdiff import DeepDiff

from ls_helper.models.interface_models import InterfaceData, ProjectFieldsExtensions
from ls_helper.new_models import ProjectData
from ls_helper.settings import SETTINGS


def get_tree_n_root(xml_file: Path) -> tuple:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return tree, root


def find_all_names(root):
    unique_names = {}

    def find_name(element, current_path):
        # Print current element's path
        path = f"{current_path}/{element.tag}"
        # print(path, element.get("name"))
        if _name := element.get("name"):
            unique_names.setdefault(_name, []).append(path)

        # Recurse through all children
        for child in element:
            find_name(child, path)

    # Start from root
    find_name(root, "")

    return unique_names


def find_tag_name_refs(root):
    refs = {}

    def find_name(element, current_path):
        # Print current element's path
        path = f"{current_path}/{element.tag}"
        # print(path, element.get("name"))
        if _name := element.get("whenTagName"):
            refs.setdefault(_name, []).append(path)

        # Recurse through all children
        for child in element:
            find_name(child, path)

    # Start from root
    find_name(root, "")

    return refs


def find_duplicates(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    unique_names = {}

    def find_name(element, current_path):
        # Print current element's path
        path = f"{current_path}/{element.tag}"
        # print(path, element.get("name"))
        if _name := element.get("name"):
            unique_names.setdefault(_name, []).append(path)

        # Recurse through all children
        for child in element:
            find_name(child, path)

    # Start from root
    find_name(root, "")

    return {
        k: v for k, v in unique_names.items() if len(v) > 1
    }


def check_references(root) -> list[str]:
    names = list(find_all_names(root).keys())
    # print(names)
    refs = list(find_tag_name_refs(root).keys())
    broken_refs = []
    for ref in refs:
        if ref not in names:
            print(ref)
            broken_refs.append(ref)
    if broken_refs:
        print("broken references:")
        for ref in broken_refs:
            print(ref)
    else:
        print("all refs ok")
    return broken_refs


def check_config_update(platform_configs: dict[str, Path]):
    for platform, fp in platform_configs.items():
        # todo, maybe diff the languages...
        next_file = SETTINGS.labeling_configs_dir / f"{platform}-next.xml"
        next_conf = ProjectData.parse_label_config_xml(fp.read_text())

        # shutil.copy(fp, next_file)

        current_file = SETTINGS.labeling_configs_dir / f"{platform}.xml"
        current_conf = ProjectData.parse_label_config_xml(current_file.read_text())

        diff = DeepDiff(current_conf, next_conf)
        print(diff)


def check_against_fixes(label_config: str | InterfaceData, fixes: ProjectFieldsExtensions):
    """
    Do
    :param label_config: xml config string
    :return:
    """
    if isinstance(label_config, str):
        conf = ProjectData.parse_label_config_xml(label_config)
    else:
        conf = label_config
    columns = set(list(conf.orig_choices.keys()) + list(conf.free_text))
    fixes_set = set(fixes.extensions)
    print(f"columns missing in fixes: {columns - fixes_set}")
    print(f"obsolete fixes: {columns - fixes_set}")


if __name__ == "__main__":
    pass
    # p_info = ProjectOverview.projects().get_project(("twitter", "en"))
    # check_against_fixes(p_info)
    """ not sure, what was done here...
    project_p = os.getcwd()
    step1_t = Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/configs/step1/output/gen_twitter.xml")
    # step1_t = Path(f"{project_p}/data//home/rsoleyma/projects/MyLabelstudioHelper/data/configs/step1/output/gen_twitter.xml")
    duplicates = find_duplicates(step1_t)
    print(f"{duplicates=}")
    # complete_config((Path("/home/rsoleyma/projects/platforms-clients/data/labelstudio_configs/final1_t/config.xml")))

    # tree = ET.parse(Path("/home/rsoleyma/projects/platforms-clients/data/labelstudio_configs/test_session_1_2025.xml"))
    tree = ET.parse(step1_t)
    root = tree.getroot()

    check_references(root)

    all_names = find_all_names(get_tree_n_root(step1_t)[1])
    all_names = list(all_names.keys())
    all_names_s = [s.split("_") for s in all_names]
    # print(json.dumps(list(zip(all_names,all_names_s)), indent=2))
    """

    # print(all_names_s)
    # fout = Path(f"{project_p}/data/extra/name_checker.csv")
    # writer = csv.writer(fout.open("w", encoding="utf-8"))
    # for s in all_names_s:
    #     # print(",".join(s))
    #     writer.writerow(s)
    #
    # print(f"-> {fout}")
