import copy
import json
import os
from pathlib import Path
from string import Template
from typing import Optional

import pystache
from lxml import etree


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


