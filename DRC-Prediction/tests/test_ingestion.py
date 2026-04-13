"""Unit tests for FEMA ingestion module."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.fema_api import _fetch_paginated


def _mock_response(data: list, key: str = "DisasterDeclarationsSummaries"):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {key: data}
    return mock


@patch("src.ingestion.fema_api.requests.get")
def test_fetch_paginated_returns_dataframe(mock_get):
    mock_get.return_value = _mock_response(
        [{"disasterNumber": 1, "state": "TX"}],
        key="DisasterDeclarationsSummaries",
    )
    from src.ingestion.fema_api import ENDPOINTS, FIELDS
    df = _fetch_paginated(
        ENDPOINTS["declarations"],
        FIELDS["declarations"],
        max_records=10,
    )
    assert isinstance(df, pd.DataFrame)


@patch("src.ingestion.fema_api.requests.get")
def test_fetch_paginated_empty_stops(mock_get):
    mock_get.return_value = _mock_response([], key="DisasterDeclarationsSummaries")
    from src.ingestion.fema_api import ENDPOINTS, FIELDS
    df = _fetch_paginated(ENDPOINTS["declarations"], FIELDS["declarations"])
    assert len(df) == 0
