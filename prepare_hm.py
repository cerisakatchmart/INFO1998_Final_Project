"""Shrink the Kaggle H&M data (~31M transaction rows) into files the pipeline can use.

Reads transactions_train.csv in chunks, aggregates to (article_id, week), keeps the
best-selling articles that have enough history, and writes:

    data/processed/hm_weekly_transactions.csv   item_id, date (week start), price, quantity
    data/processed/hm_items.csv                 item_id + a few categorical attributes

Then run:
    python -m src.run_experiment \
        --transactions data/processed/hm_weekly_transactions.csv \
        --attributes data/processed/hm_items.csv

Kaggle's `price` column is a scaled value, not dollars. Only relative price changes
matter for our features, so that is fine, but do not describe it as a dollar price.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ATTRIBUTE_COLS = [
    "product_type_name", "product_group_name", "garment_group_name",
    "perceived_colour_master_name", "index_group_name",
]


def aggregate_transactions(path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=["t_dat", "article_id", "price"],
                             dtype={"article_id": "int64", "price": "float64"},
                             parse_dates=["t_dat"], chunksize=chunksize):
        chunk["week"] = chunk["t_dat"].dt.to_period("W").dt.start_time
        chunk["revenue"] = chunk["price"]  # one row = one unit sold
        parts.append(chunk.groupby(["article_id", "week"], as_index=False)
                          .agg(quantity=("price", "size"), revenue=("revenue", "sum")))
        print(f"  processed rows through chunk {len(parts)}", flush=True)
    weekly = pd.concat(parts).groupby(["article_id", "week"], as_index=False).sum()
    weekly["price"] = weekly["revenue"] / weekly["quantity"]
    return weekly.drop(columns="revenue")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", help="folder with transactions_train.csv and articles.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--top-n", type=int, default=2000, help="keep this many best-selling articles")
    ap.add_argument("--min-active-weeks", type=int, default=26,
                    help="article must have sold in at least this many distinct weeks")
    ap.add_argument("--keep-censored", action="store_true",
                    help="also keep articles already selling in the first 4 weeks of data (true launch date unknown)")
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    weekly = aggregate_transactions(raw / "transactions_train.csv")

    stats = weekly.groupby("article_id").agg(
        units=("quantity", "sum"), active_weeks=("week", "nunique"), first_week=("week", "min"))
    keep = stats[stats["active_weeks"] >= args.min_active_weeks]
    if not args.keep_censored:
        # Articles on sale since the start of the data have an unknown launch date, so
        # "age" would be wrong for them. Dropping them biases the sample toward newer items.
        data_start = weekly["week"].min()
        keep = keep[keep["first_week"] > data_start + pd.Timedelta(weeks=4)]
    keep = keep.sort_values("units", ascending=False).head(args.top_n)
    print(f"kept {len(keep):,} of {len(stats):,} articles")

    weekly = weekly[weekly["article_id"].isin(keep.index)]
    weekly = weekly.rename(columns={"article_id": "item_id", "week": "date"})
    weekly[["item_id", "date", "price", "quantity"]].to_csv(out / "hm_weekly_transactions.csv", index=False)

    articles = pd.read_csv(raw / "articles.csv", usecols=["article_id", *ATTRIBUTE_COLS])
    articles = articles[articles["article_id"].isin(keep.index)].rename(columns={"article_id": "item_id"})
    articles.to_csv(out / "hm_items.csv", index=False)
    print(f"wrote files to {out}/")


if __name__ == "__main__":
    main()
