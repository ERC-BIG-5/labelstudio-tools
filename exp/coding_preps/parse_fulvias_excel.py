import csv
import json
from pathlib import Path

extra_base = Path("/home/rsoleyma/projects/MyLabelstudioHelper/data/extra/")
csv_file = extra_base / "data_coding_trial.csv"

reader = csv.reader(csv_file.open())
header = []
values = {}
end_h = set()

for line in reader:
    if not header and line[0] != "*":
        continue
    else:
        if not header:
            start_idx = 0
            for idx in range(len(line[1:])):
                if line[start_idx] in ["", "*"]:
                    start_idx += 1
                else:
                    break
            end_idx = line.index("*", 1)
            diff = end_idx - start_idx
            print(start_idx, end_idx)
            header = line[start_idx:end_idx]
            print(header)
            values = {h: [[]] for h in header}
            print(values)
            print(start_idx, end_idx, diff)
        # print(line)
        else:
            for idx in range(start_idx, end_idx):
                # print(idx, line[idx])
                h = header[idx - diff + 1]
                # print(idx, h)
                if line[idx] == "":
                    if values[h][-1] == []:
                        end_h.add(h)
                        continue
                    values[h].append([])
                else:
                    if h in end_h:
                        continue
                    values[h][-1].append(line[idx].strip())

for h, v in values.items():
    if v[-1] == []:
        values[h].pop(-1)

print(json.dumps(values, indent=2))
fp = extra_base / "results.json"
print(fp.as_posix())
json.dump(
    values, open(fp, "w", encoding="utf-8"), indent=2, ensure_ascii=False
)
