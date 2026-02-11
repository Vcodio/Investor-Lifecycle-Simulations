# Mortality Table (SSA-Style)

## About the bundled file

**`ssa_period_life_table.csv`** is **not** from the Social Security Administration. It was **approximated** from typical US period life table patterns so the app works out of the box. Do not treat it as official SSA data.

For results based on **official SSA data**, use the steps below.

## Official SSA data

The Social Security Administration publishes **period life tables** with death probabilities (qx) by age:

- **Main page:** https://www.ssa.gov/oact/HistEst/PerLifeTablesHome.html  
- **2024 Trustees Report (CSV downloads):** https://www.ssa.gov/oact/HistEst/PerLifeTables/2024/PerLifeTables2024.html  

Download one of the CSVs (e.g. **PerLifeTables_M_Hist_TR2024.csv** or **PerLifeTables_F_Hist_TR2024.csv**) and save it in this `data/` folder.

## Converting SSA CSV to the app format

The app expects a two-column file: `age`, `qx`. SSA files have one column per age. To convert a downloaded SSA CSV:

```bash
python data/fetch_ssa_life_table.py data/PerLifeTables_M_Hist_TR2024.csv
```

If the SSA file is in `data/` and named `PerLifeTables_M_Hist_TR2024.csv`, you can run:

```bash
python data/fetch_ssa_life_table.py
```

The script writes `data/ssa_period_life_table.csv` (using the row for year 2020 by default; edit `PICK_YEAR` in the script to change it).
