import random
from pathlib import Path

import pytest


@pytest.fixture()
def visdom_offline_logfile(dirname):

    log_file = dirname / "logs.visdom"
    yield log_file


vd_hostname = None
vd_port = None
vd_server_process = None


@pytest.fixture()
def visdom_server():
    # Start Visdom server once and stop it with visdom_server_stop
    global vd_hostname, vd_port, vd_server_process

    if vd_server_process is None:

        import subprocess
        import time

        from visdom import Visdom
        from visdom.server import download_scripts

        (Path.home() / ".visdom").mkdir(exist_ok=True)
        download_scripts()

        vd_hostname = "localhost"
        vd_port = random.randint(8089, 8887)

        try:
            vis = Visdom(server=vd_hostname, port=vd_port, raise_exceptions=True)
        except ConnectionError:
            pass

        vd_server_process = subprocess.Popen(
            ["python", "-m", "visdom.server", "--hostname", vd_hostname, "-port", str(vd_port)]
        )
        time.sleep(5)

        vis = Visdom(server=vd_hostname, port=vd_port)
        assert vis.check_connection()
        vis.close()

    yield (vd_hostname, vd_port)


@pytest.fixture()
def visdom_server_stop():

    yield None

    import time

    vd_server_process.kill()
    time.sleep(2)
