import json

d = json.load(open("scratch/task12_deep_recomputation_summary.json", encoding="utf-8"))
cm = d["confusion_matrix"]
letters = ["A", "B", "C", "D", "E", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y"]

# Header
cols = letters + ["NEUTRAL"]
header = "| Exp | " + " | ".join(cols) + " | Total |"
sep = "|:---:|" + "|".join([":---:"] * len(cols)) + "|:---:|"

print(header)
print(sep)
for row_l in letters:
    row_data = []
    tot = sum(cm[row_l].values())
    for col_l in cols:
        val = cm[row_l].get(col_l, 0)
        # Highlight non-zero
        if val > 0:
            if col_l == row_l:
                s = f"**{val}**"
            elif col_l == "NEUTRAL":
                s = f"*{val}*"
            else:
                s = f"<span style='color:red'>{val}</span>"
        else:
            s = "-"
        row_data.append(s)
    print(f"| **{row_l}** | " + " | ".join(row_data) + f" | **{tot}** |")

