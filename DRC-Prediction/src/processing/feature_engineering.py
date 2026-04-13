from pathlib import Path
import numpy as np
import pandas as pd

# map state into regions
REGION_MAP = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South",
    "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West",
    "NV": "West", "NM": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West",
}

HIGH_COST_TYPES = {"Hurricane", "Flood", "Tornado", "Severe Storm"}

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_raw(name: str) -> pd.DataFrame:
    path = DATA_RAW_DIR / f"{name}.csv"
    df = pd.read_csv(path, low_memory=False)
    return df


def aggregate_pa_obligations(pa_df: pd.DataFrame):
    """
    Aggregate project-level Public Assistance obligations to disaster level.
    Target variable: total_obligated_amount
    """
    pa_df = pa_df.copy()

    # Coerce numeric columns
    for col in ["totalObligated", "federalShareObligated", "projectAmount"]:
        if col in pa_df.columns:
            pa_df[col] = pd.to_numeric(pa_df[col], errors="coerce").fillna(0)

    # Drop records with zero or negative obligations
    pa_df = pa_df[pa_df["totalObligated"] > 0]

    agg = (
        pa_df.groupby("disasterNumber")
        .agg(
            total_obligated_amount=("totalObligated", "sum"),
            federal_share_obligated=("federalShareObligated", "sum"),
            project_count=("totalObligated", "count"),
            avg_project_amount=("projectAmount", "mean"),
        )
        .reset_index()
    )
    return agg


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def engineer_features( declarations: pd.DataFrame, pa_agg: pd.DataFrame, summaries: pd.DataFrame,) -> pd.DataFrame:
    """
    Merge datasets and engineer all model features.

    Returns:
        Processed DataFrame ready for modelling.
    """
    # Parse dates
    declarations = parse_dates(
        declarations,
        ["declarationDate", "incidentBeginDate", "incidentEndDate"],
    )

    # Incident duration (days)
    declarations["incident_duration_days"] = (
        declarations["incidentEndDate"] - declarations["incidentBeginDate"]
    ).dt.days.clip(lower=0).fillna(0)

    # Declaration year & month (seasonality)
    declarations["declaration_year"] = declarations["declarationDate"].dt.year
    declarations["declaration_month"] = declarations["declarationDate"].dt.month

    # Region encoding 
    declarations["region"] = declarations["state"].map(REGION_MAP).fillna("Other")

    # High-cost incident type flag
    declarations["is_high_cost_type"] = (
        declarations["incidentType"].isin(HIGH_COST_TYPES).astype(int)
    )

    # 5-year rolling disaster frequency per state
    declarations_sorted = declarations.sort_values(
        ["state", "declarationDate"]
    ).copy()

    freq_map = {}
    for _, grp in declarations_sorted.groupby("state"):
        grp = grp.set_index("declarationDate").sort_index()
        # Strip timezone so asof() string lookup works
        grp.index = grp.index.tz_localize(None) if grp.index.tz is None else grp.index.tz_convert(None)
        rolling_count = (
            grp["disasterNumber"]
            .resample("YE")
            .count()
            .rolling(5, min_periods=1)
            .mean()
            .rename("rolling_5yr_freq")
        )
        for row in grp.itertuples():
            year_end = str(row.Index.year) + "-12-31"
            try:
                freq_map[row.disasterNumber] = rolling_count.asof(year_end)
            except Exception:
                freq_map[row.disasterNumber] = np.nan

    declarations["rolling_5yr_freq"] = declarations["disasterNumber"].map(freq_map)
    declarations["rolling_5yr_freq"] = declarations["rolling_5yr_freq"].fillna(
        declarations["rolling_5yr_freq"].median()
    )

    # Merge PA aggregations (left join: keep all declarations)
    df = declarations.merge(pa_agg, on="disasterNumber", how="left")

    # Merge disaster summaries
    summary_cols = [
        "disasterNumber",
        "totalObligatedAmountPa",
        "totalObligatedAmountCatAb",
        "totalObligatedAmountCatC2g",
        "totalAmountIhpApproved",
    ]
    summ_cols = [c for c in summary_cols if c in summaries.columns]
    df = df.merge(summaries[summ_cols], on="disasterNumber", how="left")

    # Fallback: use totalObligatedAmountPa from summaries when PA data is missing
    if "totalObligatedAmountPa" in df.columns:
        missing_target = df["total_obligated_amount"].isna()
        fallback_vals = pd.to_numeric(df.loc[missing_target, "totalObligatedAmountPa"], errors="coerce")
        df.loc[missing_target, "total_obligated_amount"] = fallback_vals

    # Fill missing PA-derived features
    df["project_count"] = df["project_count"].fillna(0)
    df["avg_project_amount"] = df["avg_project_amount"].fillna(0)
    df["federal_share_obligated"] = df["federal_share_obligated"].fillna(0)

    # Log-transform target
    df["log_total_obligated"] = np.log1p(df["total_obligated_amount"])

    # Drop rows with missing or zero target
    df = df[df["total_obligated_amount"] > 0].dropna(subset=["total_obligated_amount"])

    return df

DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

def save_processed(df: pd.DataFrame, name: str = "processed_disasters") -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)


def run_processing() -> pd.DataFrame:
    """Full processing pipeline."""
    declarations = load_raw("declarations")
    pa = load_raw("public_assistance")
    summaries = load_raw("disaster_summaries")

    pa_agg = aggregate_pa_obligations(pa)
    df = engineer_features(declarations, pa_agg, summaries)
    save_processed(df)
    return df


if __name__ == "__main__":
    run_processing()
