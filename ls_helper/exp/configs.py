from pathlib import Path

from lxml import etree
from lxml.etree import Element

from ls_helper.models.interface_models import ProjectVariableExtensions


def find_duplicate_names(root) -> dict[str, int]:
    xpath_query = "//*[@name]"
    elements = root.xpath(xpath_query)
    name_elems = {}
    for element in elements:
        name = element.attrib["name"]
        name_elems.setdefault(name, []).append(element)

    return {n: len(elems) for n, elems in name_elems.items() if len(elems) > 1}


def find_elem_with_attribute(root, attribute_name: str) -> dict[str, Element]:
    xpath_query = f"//*[@{attribute_name}]"
    elements = root.xpath(xpath_query)
    return {e.attrib[attribute_name]: e for e in elements}


def find_elem_with_attribute_value(
    root, attribute_name: str, value: str
) -> dict[str, Element]:
    xpath_query = f"//*[@{attribute_name}='{value}']"
    elements = root.xpath(xpath_query)
    return {e.attrib[attribute_name]: e for e in elements}


def find_names(root) -> dict[str, Element]:
    return find_elem_with_attribute(root, "name")


def config_file_name_changes(
    config_path: Path, fixes: ProjectVariableExtensions, dest_path: Path
):
    tree = etree.parse(config_path)
    root = tree.getroot()
    all_names = find_names(root, "name")
    fixes_count = 0
    for name, elem in all_names.items():
        if name in fixes.extensions and (
            n_r := fixes.extensions[name].name_fix
        ):
            elem.attrib["name"] = n_r
            title_elem = list(filter(lambda n: n == f"{name}_t", all_names))
            if title_elem:
                all_names[title_elem[0]].attrib["name"] = f"{n_r}_t"
            fixes_count += 1
    print(f"Fixes {fixes_count} out of {len(all_names)}")
    print(f"result -> {dest_path}")
    dest_path.write_bytes(etree.tostring(root))
