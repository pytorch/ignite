import pytest


@pytest.fixture()
def visdom_server():

    import os
    import signal
    import subprocess
    import time

    from visdom.server import download_scripts

    download_scripts()

    hostname = "localhost"
    port = 8098
    p = subprocess.Popen("visdom --hostname {} -port {}".format(hostname, port), shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    yield (hostname, port)
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
