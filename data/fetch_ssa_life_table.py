
"""
Convert an SSA Period Life Table CSV into the app's age,qx format.

The bundled ssa_period_life_table.csv is NOT from the SSA—it is an approximation.
To use official data:

1. Download a period life table CSV from the SSA:
   https://www.ssa.gov/oact/HistEst/PerLifeTables/2024/PerLifeTables2024.html
   (e.g. PerLifeTables_M_Hist_TR2024.csv or PerLifeTables_F_Hist_TR2024.csv)

2. Save it in this directory (or pass its path as the first argument).

3. Run this script:
   python data/fetch_ssa_life_table.py [path_to_downloaded_ssa.csv]

   If no path is given, the script looks for PerLifeTables_M_Hist_TR2024.csv in data/.

Output: ssa_period_life_table.csv with columns age, qx (one row per age 0..120).
Uses the row for PICK_YEAR (default 2020) from the SSA file.
"""

import csv
import os
import sys

PICK_YEAR = 2020
DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "PerLifeTables_M_Hist_TR2024.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "ssa_period_life_table.csv")


def convert(path: str) -> int:
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
    year_col = 0

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_for_year = None
        for row in reader:
            if not row:
                continue
            try:
                y = int(row[year_col].strip())
                if y == PICK_YEAR:
                    row_for_year = row
                    break
            except (ValueError, IndexError):
                continue

    if not row_for_year:
        print(f"No row for year {PICK_YEAR}. First column of file should be year.", file=sys.stderr)
        return 1

    qx_by_age = []
    for i in range(year_col + 1, len(row_for_year)):
        try:
            q = float(row_for_year[i].strip())
        except (ValueError, IndexError):
            q = 1.0
        qx_by_age.append((len(qx_by_age), q))

    while len(qx_by_age) < 121:
        qx_by_age.append((len(qx_by_age), 1.0))
    qx_by_age = qx_by_age[:121]

    with open(OUTPUT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age", "qx"])
        for age, qx in qx_by_age:
            w.writerow([age, qx])

    print(f"Wrote {len(qx_by_age)} ages to {OUTPUT_FILE} (year {PICK_YEAR} from {os.path.basename(path)}).")
    return 0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    if not os.path.isfile(path) and path == DEFAULT_INPUT:
        print("Usage: python fetch_ssa_life_table.py [path_to_SSA_per_life_table.csv]", file=sys.stderr)
        print("Download a CSV from:", file=sys.stderr)
        print("  https://www.ssa.gov/oact/HistEst/PerLifeTables/2024/PerLifeTables2024.html", file=sys.stderr)
        print("Save it (e.g. as PerLifeTables_M_Hist_TR2024.csv in data/) then run this script.", file=sys.stderr)
        return 1
    return convert(path)


if __name__ == "__main__":
    sys.exit(main())
