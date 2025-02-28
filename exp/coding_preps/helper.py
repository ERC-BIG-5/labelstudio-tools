import json
from pathlib import Path
from typing import Optional


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


if __name__ == "__main__":
    csv_results_data = Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/extra/results.json")
    data = json.load(csv_results_data.open(encoding="utf-8"))

    print("RV options")
    rv_type_options = data["rv_type"][0]
    rv_aliases = list(map(lambda _: _.lower().replace(" ", "_"), data["rv_type"][0]))

    rv_choices = create_choice_elements(rv_type_options, rv_aliases)
    print(rv_choices)

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

    print("RV confusion TEXT")
    from string import Template

    template = Template("""<View visibleWhen="choice-selected" whenTagName="rel-value_text"
                  whenChoiceValue="$rv_alias">
                    <Text value="For the TEXT content: Can the Relational Value '$rv_name' be confused with other RVs?" name="rel-value_text_conf-$rv_alias-_t"/>
                    <Choices name="rel-value_text_conf-$rv_alias" toName="title" showInline="true" choice="multiple">
                     $choices
                     </Choices>
                </View>""")
    for rv_name, rv_alias in zip(rv_type_options, rv_aliases):

        rv_type_options_ = rv_type_options[:]
        rv_type_options_.remove(rv_name)
        rv_aliases_ = rv_aliases[:]
        rv_aliases_.remove(rv_alias)
        rv_choices = create_choice_elements(rv_type_options_, rv_aliases_)
        print(template.substitute(rv_name=rv_name, rv_alias=rv_alias, choices=rv_choices))

    print("RV confusion VISUAL")

    template = Template("""<View visibleWhen="choice-selected" whenTagName="rel-value_visual"
                  whenChoiceValue="$rv_alias">
                    <Text value="For the TEXT content: Can the Relational Value '$rv_name' be confused with other RVs?" name="rel-value_visual_conf-$rv_alias-_t"/>
                    <Choices name="rel-value_visual_conf-$rv_alias" toName="title" showInline="true" choice="multiple">
                     $choices
                     </Choices>
                </View>""")
    for rv_name, rv_alias in zip(rv_type_options, rv_aliases):

        rv_type_options_ = rv_type_options[:]
        rv_type_options_.remove(rv_name)
        rv_aliases_ = rv_aliases[:]
        rv_aliases_.remove(rv_alias)
        rv_choices = create_choice_elements(rv_type_options_, rv_aliases_)
        print(template.substitute(rv_name=rv_name, rv_alias=rv_alias, choices=rv_choices))