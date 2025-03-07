import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from pandas.core.config_init import copy_on_write_doc


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

def check_references(root):
    print("broken references:")
    names = list(find_all_names(root).keys())
    # print(names)
    refs = list(find_tag_name_refs(root).keys())
    for ref in refs:
        if ref not in names:
            print(ref)


def complete_config(xml_file: Path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img1 = root.findall(".//{*}Image")[0]
    print(img1)


if __name__ == "__main__":
    project_p = os.getcwd()
    step1_t = Path(f"{project_p}/data/configs/step1/output/gen_youtube.xml")
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

    # print(all_names_s)
    fout = Path(f"{project_p}/data/extra/name_checker.csv")
    writer = csv.writer(fout.open("w", encoding="utf-8"))
    for s in all_names_s:
        # print(",".join(s))
        writer.writerow(s)

    print(f"-> {fout}")
