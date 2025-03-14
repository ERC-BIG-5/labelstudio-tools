import json
import os
from pathlib import Path
from typing import Optional
from xml import etree

import pystache


def create_choice_elem(option: str, alias: Optional[str] = None) -> str:
    if alias:
        return f'<Choice value="{option}" alias="{alias}"/>'
    return f'<Choice value="{option}"/>'


def create_choice_elements(options: list[str], aliases: Optional[list[str]] = None) -> str:
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
    elements = root.xpath(f'//*[@className="hidden"]')

    # print(len(elements))
    # Remove all found elements from their parents
    for element in elements:
        parent = element.getparent()
        if parent is not None:  # Check if element has a parent
            parent.remove(element)

    tree.write(xml_file, pretty_print=True, encoding='utf-8', xml_declaration=False)



if __name__ == "__main__":
    csv_results_data = Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/extra/results.json")
    data = json.load(csv_results_data.open(encoding="utf-8"))

    print("RV options")
    rv_type_options = data["rv_type"][0]
    rv_aliases = list(map(lambda _: _.lower().replace(" ", "-"), data["rv_type"][0]))

    rv_choices = create_choice_elements(rv_type_options, rv_aliases)
    # XX
    # print(rv_choices)
    # print(json.dumps(data, indent=2))
    """
    
    print("NEP")
    for group in data["nep_type"]:
        type_options = group
        aliases = list(map(lambda _: _.lower().replace(" ", "_"), group))
        nep_choices = create_choice_elements(type_options, aliases)
        print(nep_choices)

    print("type of interaction")
    for group in data["i_type"]:
        type_options = group
        aliases = list(map(lambda _: _.lower().replace(" ", "_"), group))
        i_choices = create_choice_elements(type_options, aliases)
        print(i_choices)

    print("sentiment...")
    for group in data["sentiment"]:
        type_options = group
        aliases = list(map(lambda _: _.lower().replace(" ", "_"), group))
        fr_choices = create_choice_elements(type_options, aliases)
        print(fr_choices)

    print("framing")
    for group in data["framing"]:
        type_options = group
        aliases = list(map(lambda _: _.lower().replace(" ", "_"), group))
        fr_choices = create_choice_elements(type_options, aliases)
        print(fr_choices)
    """

    print("user type")
    for group in data["user_type"]:
        type_options = group
        aliases = list(map(lambda _: _.lower().replace(" ", "_"), group))
        user_choices = create_choice_elements(type_options, aliases)
        print(user_choices)
    # print("RV confusion TEXT")
    from string import Template

    template = Template("""<View visibleWhen="choice-selected" whenTagName="rel-value_text"
                  whenChoiceValue="$rv_alias">
                    <Text value="For the TEXT content: Can the Relational Value '$rv_name' be confused with other RVs?" name="rel-value_text_conf_$rv_alias-_t"/>
                    <Choices name="rel-value_text_conf_$rv_alias" toName="title" showInline="true" choice="multiple">
                     $choices
                     </Choices>
                </View>""")

    text_rv_confusion_views = ""
    for rv_name, rv_alias in zip(rv_type_options, rv_aliases):
        rv_type_options_ = rv_type_options[:]
        rv_type_options_.remove(rv_name)
        rv_aliases_ = rv_aliases[:]
        rv_aliases_.remove(rv_alias)
        rv_choices = create_choice_elements(rv_type_options_, rv_aliases_)
        text_rv_confusion_views += template.substitute(rv_name=rv_name, rv_alias=rv_alias, choices=rv_choices)

    # print(text_rv_confusion_views)
    # exit()
    # print("RV confusion VISUAL")

    template = Template("""<View visibleWhen="choice-selected" whenTagName="rel-value_visual"
                  whenChoiceValue="$rv_alias">
                    <Text value="For the TEXT content: Can the Relational Value '$rv_name' be confused with other RVs?" name="rel-value_visual_conf_-$rv_alias-_t"/>
                    <Choices name="rel-value_visual_conf_$rv_alias" toName="title" showInline="true" choice="multiple">
                     $choices
                     </Choices>
                </View>""")

    visual_rv_confusion_views = b""
    for rv_name, rv_alias in zip(rv_type_options, rv_aliases):
        rv_type_options_ = rv_type_options[:]
        rv_type_options_.remove(rv_name)
        rv_aliases_ = rv_aliases[:]
        rv_aliases_.remove(rv_alias)
        rv_choices = create_choice_elements(rv_type_options_, rv_aliases_)
        visual_rv_confusion_views += template.substitute(rv_name=rv_name, rv_alias=rv_alias, choices=rv_choices).encode(
            "utf-8")
        # print(visual_rv_confusion_views)
    # print(text_rv_confusion_views)
    base_path = Path(f"{os.getcwd()}/data/configs/step1")
    template_file = base_path / "config_template.xml"
    template_text = template_file.read_text(encoding="utf-8")

    platform_files = ["youtube.xml", "twitter.xml"]
    for platform_file in platform_files:
        platform_fp = base_path / platform_file
        platform_elements = platform_fp.read_text(encoding="utf-8")

        platform_user_file = f"{Path(platform_file).stem}_user.xml"
        platform_user_fp = base_path / platform_user_file
        if platform_user_fp.exists():
            plaform_user_elements = platform_user_fp.read_text(encoding="utf-8")
        else:
            plaform_user_elements = ""

        gen_text = pystache.render(template_text, {
            'TEXT_RV_CONFUSION': text_rv_confusion_views,
            'VISUAL_RV_CONFUSION': visual_rv_confusion_views,
            'PLATFORM_ELEMENTS': platform_elements,
            'PLATFORM_USER_ELEMENTS': plaform_user_elements
        }, encoding="utf-8")

        pl_gen_file = base_path / f"output/gen_{platform_file.split('.')[0]}.xml"
        pl_gen_file.write_text(gen_text, encoding="utf-8")
        if platform_file != "youtube.xml":
            remove_hidden_parts(pl_gen_file)
        print(f"-> {pl_gen_file}")
