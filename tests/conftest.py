from __future__ import annotations

from pathlib import Path

import pytest

from genai_recommender import datagen


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("data")
    datagen.generate(out, datagen.Scale(customers=400), seed=3)
    return out
