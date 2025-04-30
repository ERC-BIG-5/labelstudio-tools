from pydantic import BaseModel, Field

from ls_helper.models.interface_models import FieldExtension
from ls_helper.models.main_models import ProjectData


class VariableExtensionIndex(FieldExtension):
    projects: list[int] = Field(default_factory=list)


class DataExtensionIndex(BaseModel):
    projects: list[int] = Field(default_factory=list)
    variables: dict[str, VariableExtensionIndex]

    @property
    def intersection(self) -> dict[str, VariableExtensionIndex]:
        return {
            v: e
            for v, e in self.variables.items()
            if len(e.projects) == len(self.projects)
        }


def build_extension_index(projects: list[ProjectData]) -> DataExtensionIndex:
    """
    Goes through all projects fixes and build an index:
    key -> fix +  projects (which is a list of project ids)
    Currently just a simple print warning, if there is an issue.
    # todo, consider, it takes the fixes, which includes the unifixes
    :param projects:
    :return:
    """
    # annot_results = {}
    variables: dict[str, VariableExtensionIndex] = {}
    for p in projects:
        # annot_results[p.id] = create_annotations_results(p, False)
        ext = p.variable_extensions
        for k, v in ext.extensions.items():
            if k not in variables:
                variables[k] = VariableExtensionIndex.model_validate(
                    v.model_dump()
                )
                variables[k].projects.append(p.id)
            else:
                if v.model_dump() == (i_v := variables[k]).model_dump(
                        exclude={"projects"}
                ):
                    i_v.projects.append(p.id)
                else:
                    print(
                        f"WARNING: project '{p.id}:{p.alias}' has variable '{k}' differently configured as in the index\n"
                        f"({p.id:>5}): {v.model_dump()}\n"
                        f"(index): {variables[k].model_dump(exclude={'projects'})}"
                    )

    print(compare_variables(projects))
    return DataExtensionIndex(
        projects=[p.id for p in projects], variables=variables
    )


def compare_variables(projects: list[ProjectData]):
    variables: dict[str, list[int]] = {}
    for p in projects:
        # annot_results[p.id] = create_annotations_results(p, False)
        ext = p.variable_extensions
        for idx, (k, var_ext) in enumerate(ext.extensions.items()):
            name = var_ext.name_fix or k
            variables.setdefault(name, []).append(p.id)

    for v, ps in variables.items():
        if len(ps) != len(projects):
            print(f"{v}: {ps}")
    return variables
