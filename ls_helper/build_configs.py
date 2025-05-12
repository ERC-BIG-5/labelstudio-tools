from pathlib import Path
from typing import Any, Optional

import pystache
from lxml import etree
from lxml.etree import _Comment, Element
from pydantic import BaseModel, Field
from pystache.parsed import ParsedTemplate
from pystache.parser import _LiteralNode

from ls_helper.exp.configs import find_duplicate_names
from ls_helper.settings import SETTINGS
from tools.files import levenhstein_get_similar_filenames
from tools.pydantic_annotated_types import SerializablePath

SRC_COMPONENT = "src-component"


class LabelingInterfaceBuildConfig(BaseModel):
    template: SerializablePath = Field(
        ...,
        description="the fundamental template. relative to 'labelling_configs/templates'",
    )
    generic_config: Optional[dict] = Field(default_factory=dict)
    test_input_data: Optional[dict] = Field(default_factory=dict)


def validate_template_against_data(template_str: str, data: dict):
    root = etree.fromstring(template_str)
    for elem in root.iter():
        if "value" in elem.attrib:
            if (val := elem.attrib["value"]).startswith("$"):
                if val[1:] not in data:
                    print(f"{val[1:]} MISSING")


def create_choice_elem(option: str, alias: Optional[str] = None) -> str:
    if alias:
        return f'<Choice value="{option}" alias="{alias}"/>'
    return f'<Choice value="{option}"/>'


def create_choice_elements(
        options: list[str], aliases: Optional[list[str]] = None
) -> str:
    if aliases:
        res = []
        for o, a in zip(options, aliases):
            res.append(create_choice_elem(o, a))
        return "".join(res)
    return "".join(list(map(create_choice_elem, options)))


def remove_hidden_parts(xml_file: Path):
    """
    Removes all elements with a specific class attribute from an XML file using lxml.

    """
    # Parse the XML file
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Find all elements with the specified class
    # This will find elements where class attribute exactly matches the class_name
    elements = root.xpath('//*[@className="hidden"]')

    # print(len(elements))
    # Remove all found elements from their parents
    for element in elements:
        parent = element.getparent()
        if parent is not None:  # Check if element has a parent
            parent.remove(element)

    tree.write(
        xml_file, pretty_print=True, encoding="utf-8", xml_declaration=False
    )


def check_references(root) -> dict[str, list[str]]:
    names = list(find_all_names(root).keys())
    # print(names)
    refs = find_tag_name_refs(root).items()
    broken_refs = {}
    for ref_name, depending in refs:
        if ref_name not in names:
            # print(ref)
            component = None
            for d_elem in depending:
                cur_elem = d_elem
                while True:
                    comp = cur_elem.attrib.get(SRC_COMPONENT)
                    if comp:
                        component = comp
                        break
                    if not comp:
                        if (parent := cur_elem.getparent()) is None:
                            break
                        cur_elem = parent

            deps = [
                f"{d.tag}:{d.attrib.get('name', '')} [component:{component}]"
                for d in depending
            ]
            broken_refs[ref_name] = deps
    if broken_refs:
        print("broken references:")
        for ref, dep in broken_refs.items():
            print(f"{ref}:{dep}")
    else:
        print("all refs ok")
    return broken_refs


def validate_variables_against_mustache_template(
        template: ParsedTemplate, variables: dict[str, Any]
) -> tuple[set[str], list[str]]:
    """
    check if all variables in a mustache template are covered by the given variables.
    :param tempalte:
    :param variables:
    :return: a list of missing variables.
    """
    missing_variables: set[str] = set()
    literal_node_keys = []
    redundant_variables: list[str] = []
    for e in template._parse_tree:
        if isinstance(e, _LiteralNode):
            literal_node_keys.append(e.key)
            if e.key not in variables:
                missing_variables.add(e.key)
    for var in variables:
        if var not in literal_node_keys:
            redundant_variables.append(var)
    return missing_variables, redundant_variables


def build_from_template(
        config: LabelingInterfaceBuildConfig,
) -> tuple[etree.ElementTree, dict[str, list[str]], dict[str, int]]:
    """

    :param config:
    :return: the tree, broken refs (with list of of elements referring to it), duplicates (name:count)
    """

    def read_pystache2lxml_tree(
            fp: Path, attrib: dict[str, Any]
    ) -> etree.ElementTree:  # tree
        raw_text = fp.read_text(encoding="utf-8")
        template: ParsedTemplate = pystache.parse(raw_text)
        missing, redundant = validate_variables_against_mustache_template(
            template, attrib
        )
        if missing or redundant:
            print(
                f"Missing variables: {missing} / redundant variables: {redundant} for template of file: '{fp.name}'"
            )
        result = pystache.render(raw_text, context=attrib)
        try:
            tree = etree.ElementTree(etree.fromstring(result))
            return tree
        except etree.XMLSyntaxError:
            print(result)
            raise

    components_dir = SETTINGS.labeling_configs_dir / "components"

    def parse_tree(
            sub_tree: etree.ElementTree,
            parent_attrib: Optional[dict] = None,
            parent_slot_fillers: Optional[list[etree.Element]] = (),
    ) -> etree.Element:
        """

        :param sub_tree:
        :param parent_attrib:
        :param parent_slot_fillers:
        :return:
        """
        if not parent_attrib:
            parent_attrib = {}

        nodes_to_process = list(sub_tree.getroot().iter())
        for node in nodes_to_process:
            # some LSElements and basic html elements to ignore
            if node.tag in [
                "Style",
                "Collapse",
                "Panel",
                "Choices",
                "Header",
                "Text",
                "Image",
                "TextArea",
                "Choice",
                "Video",
                "HyperText",
                "Label",
                "TimelineLabels",
                "a",
                "div",
            ]:
                continue
            # print(node.tag)
            if isinstance(node, _Comment):
                continue

            if node.tag == "View":
                if (_if := node.attrib.get("if")) and (
                        _is := node.attrib.get("is")
                ):
                    del node.attrib["if"]
                    del node.attrib["is"]
                    node.attrib.update(
                        {
                            "visibleWhen": "choice-selected",
                            "whenTagName": _if,
                            "whenChoiceValue": _is,
                        }
                    )
                # todo, this is an infinite loop. but all we want is process internal nodes...
                """
                view_node = parse_tree(
                    etree.ElementTree(node), parent_attrib, parent_slot_fillers
                )
                node.getparent().replace(node, view_node)"""
                continue

            if node.tag == "slot":
                # todo, for now, just take the first
                if parent_slot_fillers:
                    node.getparent().replace(node, parent_slot_fillers[0])  #

                else:
                    node.getparent().remove(node)
                continue

            src_file = components_dir / f"{node.tag}.xml"
            # print(src_file)

            if not src_file.exists():
                try:
                    levenhstein_sim = levenhstein_get_similar_filenames(
                        str(node.tag), components_dir
                    )
                except ImportError:
                    levenhstein_sim = "... no 'python-Levenshtein' installed. so no similarity for you"
                print(
                    f"file for component: '{node.tag}' not found, maybe: {levenhstein_sim}"
                )
                continue

            # parse slot-filler elements and collect
            slot_filler_nodes = []
            for slot_elem in node.getchildren():
                slot_filler_nodes.append(
                    parse_tree(etree.ElementTree(slot_elem), parent_attrib)
                )

            # merge attributes
            updated_attrib = parent_attrib | dict(node.attrib)
            # read file and do mustache rendering
            component_tree = read_pystache2lxml_tree(src_file, updated_attrib)
            # recursive tree parsing
            component_node = parse_tree(
                component_tree, updated_attrib, slot_filler_nodes
            )
            # add metadata
            comment = etree.Comment(f"component:{node.tag}")
            component_node.insert(
                0, comment
            )  # 1 is the index where comment is inserted
            component_node.attrib[SRC_COMPONENT] = node.tag

            node.getparent().replace(node, component_node)
        return sub_tree.getroot()

    _tree: etree.ElementTree = read_pystache2lxml_tree(config.template, {})
    parse_tree(_tree)
    root = _tree.getroot()

    broken_refs = check_references(root)
    dupl_names = find_duplicate_names(root)
    print(f"duplicate name: {dupl_names}")

    return _tree, broken_refs, dupl_names


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


def find_tag_name_refs(root) -> dict[str, list[Element]]:
    """
    checks which elements depend on all variables
    :param root:
    :return: dict: variable: [depending_0, depending_1, ...]
    """
    refs: dict[str, list[str]] = {}

    def find_name(element, current_path):
        # Print current element's path
        path = f"{current_path}/{element.tag}"
        # print(path, element.get("name"))
        if _name := element.get("whenTagName"):
            refs.setdefault(_name, []).append(element)

        # Recurse through all children
        for child in element:
            find_name(child, path)

    # Start from root
    find_name(root, "")

    return refs
