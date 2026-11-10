import json

with open("backend/data/sources/qipedc_dictionary.json", "r", encoding="utf-8") as f:
    q_dict = json.load(f)

for row in q_dict:
    vid = row[0]
    word = row[1]
    if any(row[0].startswith(prefix) for prefix in ["D050", "D051", "D052"]) or word.upper() in ["P", "Q", "R", "X", "CHỮ P", "CHỮ Q", "CHỮ R", "CHỮ X"]:
        print(f"VID: {vid:10s} | Word: {word}")
