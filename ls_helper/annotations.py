import json
from pathlib import Path

from ls_helper.models import ProjectAnnotationExtension
from ls_helper.settings import SETTINGS


def get_platform_fixes(project_id: int) -> ProjectAnnotationExtension:
    """
    get the annotation-fxies for a project.

    """
    if (fi := SETTINGS.fixes_dir / "unifixes.json").exists():
        data_extensions = ProjectAnnotationExtension.model_validate(json.load(fi.open()))
    else:
        print(f"no unifixes.json file yet in {SETTINGS.fixes_dir / 'unifix.json'}")
        data_extensions = {}
    if (fi := SETTINGS.BASE_DATA_DIR / f"fixes/{project_id}.json").exists():
        p_fixes = ProjectAnnotationExtension.model_validate(json.load(fi.open()))
        data_extensions.project_id = project_id
        data_extensions.fixes.update(p_fixes.fixes)
        data_extensions.fix_reverse_map.update(p_fixes.fix_reverse_map)

    return data_extensions


if __name__ == "__main__":
    uni = ProjectAnnotationExtension.model_validate(json.load(SETTINGS.unifix_file_path.open()))
    twitter = ProjectAnnotationExtension.model_validate(json.load((SETTINGS.fixes_dir / "39.json").open()))
    yt = ProjectAnnotationExtension.model_validate(json.load((SETTINGS.fixes_dir / "33.json").open()))
    uni_f = set(uni.fixes.keys())
    twitter_f = set(twitter.fixes.keys())
    yt_t = set(yt.fix_reverse_map.keys())

    """ merge to create unifixes DONE
    merge = {}
    print(yt_t -  twitter_f)
    print(twitter_f - yt_t)
    i_s = yt_t.intersection(twitter_f)
    print(i_s)
    for k in i_s:
        print(k)
        if yt.fixes[k].name_fix != twitter.fixes[k].name_fix:
            print(k, yt.fixes[k].name_fix,twitter.fixes[k].name_fix )

        merge[k] = yt.fixes[k]

    merge = {k: merge[k] for k in yt.fixes.keys() if k in i_s}

    #SETTINGS.unifix_file_path.write_text(ProjectAnnotationExtension(project_id=0, fixes=merge).model_dump_json(indent=2))
    """

    """ removed the fixes in the platforms which are in the unifix DONE
    print("tw")
    for k in [k for k in twitter.fixes.keys() if k in uni_f]:
        print(k)
        del twitter.fixes[k]

    (SETTINGS.fixes_dir /  "39.json").write_text(twitter.model_dump_json(indent=2, exclude_none=True))
    print("-------")
    print("yt")
    for k in [k for k in yt.fixes.keys() if k in uni_f]:
        print(k)
        del yt.fixes[k]

    #(SETTINGS.fixes_dir / "33.json").write_text(yt.model_dump_json(indent=2, exclude_none=True))
    """