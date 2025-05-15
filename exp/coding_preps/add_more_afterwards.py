"""
push from twitter-en-3 to twitter-en-5, but only those which are not yet included
"""

from ls_helper.models.main_models import get_project
from ls_helper.settings import SETTINGS
from tools.files import read_data

po = get_project(id=43)
cleanies = read_data(po.path_for(
    SETTINGS.annotations_dir,
    alternative=f"clean_{po.id}",
))

new_po = get_project(53)
tasks = new_po.get_tasks()
rel = set()
for p_id, ann in cleanies.items():
    for a in ann:
        try:
            if a["nature_any"] == "Yes":
                rel.add(p_id)
                # print(p_id)
        except:
            print(f"not ann.{p_id}. {ann}")

print(len(rel))

all_ex = [new_p.data["platform_id"] for new_p in tasks.root]
missing = rel - set(all_ex)

old_tasks = po.get_tasks()
new_add_tasks = [t for t in old_tasks.root if t.data["platform_id"] in missing]
for nt in new_add_tasks[1:]:
    pass
    # nt.project = new_po.id
    # re = ls_client().create_task(nt)
