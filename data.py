"""Load transactions into a weekly (item, week) panel, plus a synthetic demo generator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_weekly_panel(
    transactions: pd.DataFrame,
    attributes: pd.DataFrame | None = None,
    item_col: str = "item_id",
    date_col: str = "date",
    price_col: str = "price",
    qty_col: str | None = "quantity",
) -> pd.DataFrame:
    """Aggregate raw transactions to one row per (item_id, week).

    Output columns: item_id, week, units, price, launch_week, plus any attribute columns.
    Weeks after an item's first sale with no sales are filled with 0 units (price carried
    forward), so lag features are not silently skipping weeks.
    """
    df = transactions.rename(columns={item_col: "item_id", date_col: "date", price_col: "price"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    if qty_col and qty_col in df.columns:
        df = df.rename(columns={qty_col: "quantity"})
    else:
        df["quantity"] = 1
    df["week"] = df["date"].dt.to_period("W").dt.start_time

    df["revenue"] = df["price"] * df["quantity"]
    weekly = df.groupby(["item_id", "week"], as_index=False).agg(
        units=("quantity", "sum"), revenue=("revenue", "sum")
    )
    weekly["price"] = weekly["revenue"] / weekly["units"]
    weekly = weekly.drop(columns="revenue")

    # Give every item a row for every week from its first sale to the LAST week in the data.
    # Zero-sales weeks (including after an item stops selling) are real zeros, not missing rows;
    # dropping them would hide exactly the "sales fall off" cases a forecaster must handle.
    first = weekly.groupby("item_id")["week"].min()
    last_week = weekly["week"].max()
    n_weeks = ((last_week - first).dt.days // 7 + 1).to_numpy()
    offsets = np.concatenate([np.arange(n) for n in n_weeks])
    grid = pd.DataFrame({
        "item_id": np.repeat(first.index.to_numpy(), n_weeks),
        "week": np.repeat(first.to_numpy(), n_weeks) + offsets.astype("timedelta64[W]"),
    })
    panel = grid.merge(weekly, on=["item_id", "week"], how="left")
    panel["units"] = panel["units"].fillna(0)
    panel["price"] = panel.groupby("item_id")["price"].ffill()

    panel["launch_week"] = panel["item_id"].map(first)

    if attributes is not None:
        attrs = attributes.rename(columns={item_col: "item_id"})
        panel = panel.merge(attrs, on="item_id", how="left")
    return panel.sort_values(["item_id", "week"]).reset_index(drop=True)


def make_synthetic_panel(n_items: int = 150, n_weeks: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic weekly panel for smoke-testing the pipeline ONLY (not real data)."""
    rng = np.random.default_rng(seed)
    categories = ["tops", "dresses", "outerwear", "pants"]
    colors = ["black", "white", "blue", "red", "green"]
    start = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n_items):
        cat = rng.choice(categories)
        color = rng.choice(colors)
        launch = int(rng.integers(0, n_weeks // 2))
        base = rng.uniform(5, 40)
        list_price = rng.uniform(15, 120)
        price = list_price
        for w in range(launch, n_weeks):
            age = w - launch
            # occasional markdowns
            if rng.random() < 0.08:
                price = list_price * rng.uniform(0.6, 0.9)
            elif rng.random() < 0.3:
                price = list_price
            season = 1 + (0.4 if cat == "outerwear" else 0.2) * np.cos(2 * np.pi * (w % 52 - 5) / 52)
            decay = np.exp(-age / 30)
            elasticity = (list_price / price) ** 1.5
            mu = base * season * decay * elasticity
            units = rng.poisson(max(mu, 0.1))
            rows.append(
                dict(item_id=f"item_{i}", week=start + pd.Timedelta(weeks=w), units=units,
                     price=round(price, 2), launch_week=start + pd.Timedelta(weeks=launch),
                     category=cat, color=color)
            )
    return pd.DataFrame(rows)
