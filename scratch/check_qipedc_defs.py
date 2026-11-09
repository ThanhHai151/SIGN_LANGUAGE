import json

with open("backend/data/sources/qipedc_dictionary.json", "r", encoding="utf-8") as f:
    q_dict = json.load(f)

for row in q_dict:
    vid = row[0]
    if vid in ["D0509B", "D0509N", "D0509T", "D0510", "D0511", "D0517B", "D0517N", "D0517T"]:
        print(f"VID: {vid:10s} | Word: {row[1]} | Pos: {row[2]} | Def: {row[3]}")
