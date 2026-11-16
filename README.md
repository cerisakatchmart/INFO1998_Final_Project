# Predicting Next-Week Sales of Clothing Items

This is the repository for our final project in INFO1998 (Cornell Data Science).

## Hypothesis

A product's recent sales, age, price changes, and attributes together predict its
next-week sales more accurately than its recent sales alone.

## Research Question

Is a predictive model better at forecasting next-week sales than just basic trend baselines alone?

## What We Did

This is a **regression** problem: for each clothing item and each week, we predict how many
units it will sell the following week.

1. Loaded the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview)
   purchase data and counted units sold per item per week.
2. Explored and visualized the data.
3. Built features: recent sales (lags, 4-week average), price change, item age, season, and attributes.
4. Compared simple **baselines** (last week's sales, 4-week average) with **linear regression** and
   **random forest** models.
5. For each model type, compared a **sales-only** version against a **full-features** version.
   Only the features differ, so any gain comes from the extra features, which is our hypothesis.
6. Validated with a time-ordered train/test split (no shuffling) and time-series cross-validation.

Everything is in one notebook: [`notebooks/project.ipynb`](notebooks/project.ipynb).

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── project.ipynb     # the whole project, start here
├── src/
│   ├── prepare_hm.py     # shrinks the huge Kaggle file into a weekly file
│   └── data.py           # builds the weekly table; fake demo data
└── data/
    ├── raw/              # downloaded Kaggle files (not committed)
    └── processed/        # output of prepare_hm.py (not committed)
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/project.ipynb
```

With no data downloaded, the notebook runs on **fake demo data** and prints a warning. That is
only for checking that everything works. Do not report demo results.

To use the real data:

1. Join the Kaggle competition and accept its rules (read them to confirm how the data may be used).
2. Download `transactions_train.csv` and `articles.csv` into `data/raw/`
   (website download, or the Kaggle command line: `kaggle competitions download -c h-and-m-personalized-fashion-recommendations`).
3. Run `python -m src.prepare_hm` from the repo root. This creates two small files in `data/processed/`.
4. Run the notebook again. It detects the files and uses them.

## Notes on the Data (mention in the write-up)

- The raw transactions file is about 31 million rows, so `prepare_hm.py` keeps only the 2,000
  best-selling articles that sold in at least 26 different weeks. Our results apply to
  **popular** items, not the whole catalog.
- Articles already selling in the first 4 weeks of the data are dropped, since their launch date is
  unknown and "age" would be wrong for them. This biases the sample toward newer items.
  (`--keep-censored` keeps them.)
- Kaggle's `price` is a scaled value, not dollars. The model uses it as a price *level* and as a
  week-over-week % change. Both are fine because the scaling is the same for every item, but don't
  report prices in dollars.
- Weekly price is the average over online and in-store sales, so a change in the online/store mix
  can look like a small price change.
- Transactions run from Thu 2018-09-20 to Tue 2020-09-22. Weeks run Monday to Sunday, and the
  partial first and last weeks are dropped, so the panel covers 2018-09-24 to 2020-09-20.
- The data includes spring 2020 (COVID store closures), which is an unusual period for sales.
- `prepare_hm.py` was tested on a fake file with the same format and date range, not the real
  download. If it runs out of memory, lower `chunksize` in the script.

## Team

Cerisa Katchmart, ck865
Lara De, ld569
