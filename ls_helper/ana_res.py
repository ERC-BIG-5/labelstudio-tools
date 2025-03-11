import xml.etree.ElementTree as ET
from csv import DictWriter
from pathlib import Path
from typing import Optional

from ls_helper.models import ResultStruct, Choices, Choice, VariableExtension, TaskResultModel, TaskAnnotResults, \
    ProjectAnnotations, ProjectAnnotationExtension, ProjectAnnotationResults


def get_config_project_project_data(project_data: dict):
    return project_data["label_config"]


def parse_label_config_xml(xml_string,
                           project_id: int,
                           include_text: bool = False,
                           include_text_names: Optional[list[str]] = None) -> ResultStruct:
    root: ET.Element = ET.fromstring(xml_string)

    ordered_fields: list[str] = []
    choices = {}
    free_text = []
    variable_text_fields: dict[str, str] = {}  # New list for text fields with "$" values

    for el in root.iter():
        if el.tag == "Choices":
            name = el.get('name')
            ordered_fields.append(name)
            # print(choices_element.attrib)
            choice_list = [Choice.model_validate(choice.attrib) for choice in el.findall('./Choice')]
            choices[name] = Choices.model_validate(el.attrib | {"options": choice_list})
        elif el.tag == "TextArea":
            name = el.get('name')
            free_text.append(name)
            ordered_fields.append(name)
        elif el.tag == "Text" and include_text:
            value = el.get('value')
            if value and value.startswith('$'):
                name = el.get('name')
                if not include_text_names or name in include_text_names:
                    variable_text_fields[name] = value[1:]
                    ordered_fields.append(name)

    return ResultStruct(
        project_id=project_id,
        ordered_fields=ordered_fields,
        choices=choices,
        free_text=free_text,
        inputs=variable_text_fields)



