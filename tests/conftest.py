import pytest

from mindclassify.config import Settings
from mindclassify.pipeline import prepare
from mindclassify.synthetic import SynthSpec


@pytest.fixture(scope="session")
def settings():
    return Settings(seed=3)


@pytest.fixture(scope="session")
def prepared(settings):
    return prepare(settings, synthetic=True, spec=SynthSpec(n=2500, seed=3))
