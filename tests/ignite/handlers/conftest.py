import subprocess
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from visdom import Visdom
from visdom.server.build import download_scripts


@pytest.fixture(scope="session")
def visdom_server():
    # Start Visdom server once and stop it with visdom_server_stop
    vd_hostname = "localhost"
    if not (Path.home() / ".visdom").exists():
        (Path.home() / ".visdom").mkdir(exist_ok=True)
        download_scripts()
    vis = None

    vd_port = 29777
    vd_server_process = subprocess.Popen(
        ["python", "-m", "visdom.server", "--hostname", vd_hostname, "-port", str(vd_port)]
    )
    time.sleep(2)
    for ii in range(5):
        try:
            time.sleep(1)
            vis = Visdom(server=vd_hostname, port=vd_port, raise_exceptions=True)
            break
        except ConnectionError:
            continue

    assert vis and vis.check_connection()
    yield (vd_hostname, vd_port)
    # Trying to clean up slows things down and sometimes causes hangs.
    # vis.close()
    # vd_server_process.kill()


@pytest.fixture
def no_site_packages(request):
    import sys

    modules = {}
    for k in sys.modules:
        if request.param in k:
            modules[k] = sys.modules[k]
    for k in modules:
        del sys.modules[k]

    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    for k in modules:
        sys.modules[k] = modules[k]


@pytest.fixture()
def norm_mock():
    def norm(x: torch.Tensor):
        return x.norm()

    norm_mock = Mock(side_effect=norm, spec=norm)
    norm_mock.configure_mock(**{"__name__": "norm"})
    norm_mock.reset_mock()
    return norm_mock


@pytest.fixture()
def dummy_model_factory():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(12, 12)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.fill_(1.0)
            self.fc2.bias.data.fill_(1.0)

    def get_dummy_model(with_grads=True, with_frozen_layer=False, with_buffer=False):
        model = DummyModel()
        if with_grads:
            model.fc2.weight.grad = torch.zeros_like(model.fc2.weight)
            model.fc2.bias.grad = torch.zeros_like(model.fc2.bias)

            if not with_frozen_layer:
                model.fc1.weight.grad = torch.zeros_like(model.fc1.weight)
                model.fc1.bias.grad = torch.zeros_like(model.fc1.bias)

        if with_frozen_layer:
            for param in model.fc1.parameters():
                param.requires_grad = False

        if with_buffer:
            model.register_buffer("buffer1", torch.ones(1))
        return model

    return get_dummy_model
