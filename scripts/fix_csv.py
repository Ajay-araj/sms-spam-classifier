import csv
from pathlib import Path

in_path = Path("data/spam.csv")
out_path = Path("data/spam_fixed.csv")

with in_path.open("r", encoding="latin-1") as f_in, out_path.open("w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    # write a safe header
    writer.writerow(["label", "text"])
    for i, raw in enumerate(f_in, start=1):
        line = raw.rstrip("\n\r")
        if i == 1 and (line.lower().startswith("label") or line.lower().startswith("v1")):
            # skip original header (we already wrote our header)
            continue
        parts = line.split(",", 1)   # SPLIT ON FIRST COMMA ONLY
        if len(parts) == 2:
            label, text = parts
            label = label.strip()
            text = text.strip()
            writer.writerow([label, text])
        else:
            # fallback: empty label, whole line as text
            writer.writerow(["", line.strip()])

print("Cleaned CSV written to:", out_path)
