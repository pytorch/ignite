import pytest


@pytest.fixture()
def visdom_offline_logfile(dirname):
    log_file = dirname / "logs.visdom"
    yield log_file
