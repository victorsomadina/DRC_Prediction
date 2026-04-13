import time
from pathlib import Path
import pandas as pd
import requests
from src.utils.helpers import get_logger

log = get_logger(__name__)

BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "declarations": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

FIELDS = {
    "declarations": [
        "disasterNumber", "state", "declarationType", "incidentType",
        "declarationDate", "incidentBeginDate", "incidentEndDate",
        "fyDeclared", "designatedArea", "declarationRequestNumber",
    ],
    "public_assistance": [
        "disasterNumber", "stateAbbreviation", "incidentType",
        "projectAmount", "federalShareObligated", "totalObligated",
        "projectSize", "damageCategoryCode", "applicationTitle",
    ],
    "disaster_summaries": [
        "disasterNumber", "totalAmountIhpApproved", "totalAmountHaApproved",
        "totalAmountOnaApproved", "totalObligatedAmountPa",
        "totalObligatedAmountCatAb", "totalObligatedAmountCatC2g",
        "totalObligatedAmountHmgp", "paLoadDate", "iaLoadDate",
    ],
}

def _fetch_paginated(endpoint: str, fields: list[str], max_records: int | None = None):
    records = []
    skip = 0
    params = {
        "$format": "json",
        "$top": 1000,
        "$select": ",".join(fields), # convert into comma separated value
        "$inlinecount": "allpages",
    }

    while True:
        resp = requests.get(endpoint, params={**params, "$skip": skip}, timeout=60)
        resp.raise_for_status()

        payload = resp.json()
        batch = payload[[k for k in payload if k not in ("metadata", "count")][0]]

        if not batch:
            break

        records.extend(batch)
        skip = skip + 1000
        log.info("fetched %d records so far", len(records))

        if (max_records and len(records) >= max_records) or len(batch) < 1000:
            break

        time.sleep(3.0) # to pause for 3 seconds between requests 

    if max_records:
        records = records[:max_records]

    return pd.DataFrame(records)


def fetch_declarations(max_records: int | None = None) -> pd.DataFrame:
    return _fetch_paginated(ENDPOINTS["declarations"], FIELDS["declarations"], max_records)


def fetch_public_assistance(max_records: int | None = None) -> pd.DataFrame:
    return _fetch_paginated(ENDPOINTS["public_assistance"], FIELDS["public_assistance"], max_records)


def fetch_disaster_summaries(max_records: int | None = None) -> pd.DataFrame:
    return _fetch_paginated(ENDPOINTS["disaster_summaries"], FIELDS["disaster_summaries"], max_records)


DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def save_raw(df: pd.DataFrame, name: str):
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def run_ingestion(max_records: int | None = None) -> dict[str, pd.DataFrame]:
    datasets = {
        "declarations": fetch_declarations(max_records),
        "public_assistance": fetch_public_assistance(max_records),
        "disaster_summaries": fetch_disaster_summaries(max_records),
    }
    for name, df in datasets.items():
        save_raw(df, name)
    return datasets


if __name__ == "__main__":
    run_ingestion()
