import xml.etree.ElementTree as ET
from pathlib import Path

from deepdiff import DeepDiff

from ls_helper.models.interface_models import (
    IChoice,
    IChoices,
    IField,
    InterfaceData,
    IText,
    ITextArea,
    ProjectVariableExtensions,
    ITimelineLabel,
    ILabel,
)

# from ls_helper.new_models import ProjectData
from ls_helper.settings import SETTINGS


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

    return {k: v for k, v in unique_names.items() if len(v) > 1}


def parse_label_config_xml(xml_string) -> InterfaceData:
    root: ET.Element = ET.fromstring(xml_string)

    ordered_fields_map: dict[str, IField] = {}
    # todo why this?
    input_text_fields: dict[
        str, str
    ] = {}  # New list for text fields with "$" values
    # text_areas = dict[str,ITextArea]

    for el in root.iter():
        if el.tag == "Choices":
            name = el.get("name")
            if not name:
                raise ValueError("shouldnt happen in a valid interface")
            # print(choices_element.attrib)
            choice_list = [
                IChoice.model_validate(choice.attrib)
                for choice in el.findall("./Choice")
            ]
            ordered_fields_map[name] = IChoices.model_validate(
                el.attrib
                | {
                    "options": choice_list,
                    "required": el.attrib.get("required", False),
                }
            )
        elif el.tag == "TextArea":
            name = el.get("name")
            ordered_fields_map[name] = ITextArea(
                name=name, required=el.attrib.get("required", False)
            )
        elif el.tag == "Text":
            value = el.get("value")
            if value and value.startswith("$"):
                name = el.get("name")
                # keep the $ so we know its a ref to data. done,
                input_text_fields[name] = value
                ordered_fields_map[name] = IText()
        elif el.tag == "TimelineLabels":
            name = el.get("name")
            label_list = [
                ILabel.model_validate(choice.attrib)
                for choice in el.findall("./Label")
            ]
            ordered_fields_map[name] = ITimelineLabel.model_validate(
                el.attrib | {"labels": label_list}
            )
        elif el.tag in [
            "Choice",
            "View",
            "Header",
            "Panel",
            "Collapse",
            "Image",
            "Style",
            "Video",
            "a",
            "Label",
            "HyperText",
        ]:
            pass
        else:
            print(f"parse_label_config_xml: Unknown Element-tag {el.tag}")

    return InterfaceData(
        ordered_fields_map=ordered_fields_map, inputs=input_text_fields
    )


def check_config_update(platform_configs: dict[str, Path]):
    for platform, fp in platform_configs.items():
        # todo, maybe diff the languages...
        _ = SETTINGS.labeling_configs_dir / f"{platform}-next.xml"
        next_conf = parse_label_config_xml(fp.read_text())

        # shutil.copy(fp, next_file)

        current_file = SETTINGS.labeling_configs_dir / f"{platform}.xml"
        current_conf = parse_label_config_xml(current_file.read_text())

        diff = DeepDiff(current_conf, next_conf)
        print(diff)


def check_against_fixes(
    label_config: str | InterfaceData, fixes: ProjectVariableExtensions
):
    """
    Do
    :param label_config: xml config string
    :return:
    """
    if isinstance(label_config, str):
        conf = parse_label_config_xml(label_config)
    else:
        conf = label_config
    columns = set(list(conf.orig_choices.keys()) + list(conf.free_text))
    fixes_set = set(fixes.extensions)
    print(f"columns missing in fixes: {columns - fixes_set}")
    print(f"obsolete fixes: {columns - fixes_set}")


if __name__ == "__main__":
    pass
