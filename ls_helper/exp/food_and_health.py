from csv import DictWriter

from ls_helper.command import annotations
from ls_helper.models.main_models import get_project
from ls_helper.settings import SETTINGS

if __name__ == "__main__":
    # project 50 (youtube-en) is not completed yes
    writer: dict[str, DictWriter] = {}
    vars = ["Food", "Health"]
    for d in vars:
        writer[d] = DictWriter(
            (SETTINGS.temp_file_path / f"extra_{d}.csv").open("w"),
            fieldnames=["project", "platform_id"],
        )
        writer[d].writeheader()

    for i in [43, 50, 51]:  # tw-en,yt-en, tw-es,
        po = get_project(i)
        f, res = annotations.clean_results(i, variables={"extras"})
        # print(f)
        for p_id, coder_res in res.items():
            found_extra = []
            for res in coder_res:
                for d in vars:
                    if d in found_extra:
                        continue
                    if d in res.get("extras", []):
                        writer[d].writerow(
                            {
                                "project": f"{po.platform}/{po.language}",
                                "platform_id": p_id,
                            }
                        )
                        found_extra.append(d)
