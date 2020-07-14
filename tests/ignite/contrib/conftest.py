import pytest


@pytest.fixture()
def visdom_server():

    # import os
    # import signal
    import subprocess
    import time

    from visdom import Visdom
    from visdom.server import download_scripts

    download_scripts()

    hostname = "localhost"
    port = 8098

    # vis = Visdom(server=hostname, port=port)
    # assert not vis.check_connection()

    # p = subprocess.Popen("visdom --hostname {} -port {}".format(hostname, port), shell=True, preexec_fn=os.setsid)
    p = subprocess.Popen(["python", "-m", "visdom.server", "--hostname", hostname, "-port", str(port)])
    time.sleep(5)
    yield (hostname, port)
    # os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    p.kill()

