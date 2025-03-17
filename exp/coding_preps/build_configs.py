import copy
import json
import os
from pathlib import Path
from typing import Optional
from lxml import etree
from string import Template
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


def twitter_visual_mod(xml_file: Path):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Find all elements with the specified class
    # This will find elements where class attribute exactly matches the class_name
    elements = root.xpath(f'.//*[@idAttr="visual_part"]')
    # print(len(elements))
    assert len(elements) == 1
    element = elements[0]
    element.set('className', 'hidden')

    new_template = Template("""
    <View idAttr="visual_part" visibleWhen="choice-selected" whenTagName="nature_visual" whenChoiceValue="Yes">
    <Collapse>
        $PANELS
     </Collapse>
    </View>
    """)

    make_single_select = [ "nep_group-0_visual", "nep_group-1_visual", "nep_group-2_visual",
                          "nep_group-2x_visual", "inter_group-0_visual", "inter_group-1_visual","inter_group-2_visual"]
    fix_whenTagName = ['val-expr_visual','rel-value_visual', 'nep_group-2_visual', 'inter_group-0_visual', 'inter_group-1_visual', 'inter_group-2_visual']

    panels = []
    for i in range(4):
        img_node = copy.deepcopy(element)
        #del img_node.attrib["visibleWhen"]
        img_node.set("whenTagName","media_relevant")
        img_node.set("whenChoiceValue",f"Image {i+1}")
        del img_node.attrib["className"]
        name_elements_dict = {el.get('name'): el for el in img_node.xpath('.//*[@name]')}
        # print(name_elements_dict.keys())
        panel = etree.Element('Panel')
        panel.set("value", f"Image: {i + 1}")
        for name, elem in name_elements_dict.items():
            assert "_visual" in name
            elem.set('name', name.replace("_visual", f"_visual-{i}"))
            if name in make_single_select:
                assert elem.get("choice") == "multiple", f"{name} has {elem.get('choice')}"
                elem.set("choice","single")

        #
        fix_whenTagName_elems  =  img_node.xpath('.//*[@whenTagName]')
        # print(len(fix_whenTagName_elems))
        for elem in fix_whenTagName_elems:
            whenTagName = elem.get('whenTagName')
            assert "_visual" in whenTagName
            # print(name)
            if whenTagName not in fix_whenTagName: # ignore: 'nature_visual'
                continue
            elem.set('whenTagName', whenTagName.replace("_visual", f"_visual-{i}"))
        #

        header = img_node.xpath('./*[@className="visual_part_header"]')
        assert len(header) == 1
        header[0].getparent().remove(header[0])
        # assert len(header) == 1
        panel.append(img_node)
        panels.append(panel)

    new_visual = new_template.substitute(PANELS="".join([etree.tostring(p, encoding="unicode") for p in panels]))
    # print(new_visual)
    new_visual_elem = etree.fromstring(new_visual)
    # element.getparent().replace(element, new_visual_elem)
    element.addnext(new_visual_elem)
    element.set('className', 'hidden2')
    # for i, s_elem in enumerate(element.getparent().getchildren()):
    #     if element == s_elem:
    #         s_elem.getparent().replace(s_elem, new_visual_elem)
    #         break

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

        if platform_file == "twitter.xml":
            twitter_visual_mod(pl_gen_file)
