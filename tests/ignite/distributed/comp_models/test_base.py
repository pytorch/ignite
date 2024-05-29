import pytest
import torch

from ignite.distributed.comp_models.base import _SerialModel, _torch_version_gt_112, ComputationModel


def test_serial_model():
    _SerialModel.create_from_backend()
    model = _SerialModel.create_from_context()

    assert model.get_local_rank() == 0
    assert model.get_rank() == 0
    assert model.get_world_size() == 1
    assert model.get_nproc_per_node() == 1
    assert model.get_nnodes() == 1
    assert model.get_node_rank() == 0
    if torch.cuda.is_available():
        assert model.device().type == "cuda"
    elif _torch_version_gt_112 and torch.backends.mps.is_available():
        assert model.device().type == "mps"
    else:
        assert model.device().type == "cpu"
    assert model.backend() is None
    model.finalize()

    with pytest.raises(NotImplementedError, match=r"Serial computation model does not implement spawn method"):
        model.spawn()

    model.all_reduce(1)
    model.all_gather(1)
    model.broadcast(1)
    assert model._do_all_reduce(torch.tensor(1)) == torch.tensor(1)
    assert model._do_all_gather(torch.tensor(1)) == torch.tensor(1)
    assert model._do_broadcast(torch.tensor(1), src=0) == torch.tensor(1)
    model.barrier()


def test__encode_str__decode_str():
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    s = "test-abcedfg"

    encoded_s = ComputationModel._encode_str(s, device, 1024)
    assert isinstance(encoded_s, torch.Tensor) and encoded_s.shape == (1, 1025)

    decoded_s = ComputationModel._decode_str(encoded_s)
    assert isinstance(decoded_s, list) and len(decoded_s) == 1
    assert decoded_s[0] == s


def test__encode_input_data():
    encoded_msg = ComputationModel._encode_input_data(None, is_src=True)
    assert encoded_msg == [-1] * 512

    encoded_msg = ComputationModel._encode_input_data(12.0, is_src=True)
    assert encoded_msg == [1] + [-1] * 511

    encoded_msg = ComputationModel._encode_input_data("abc", is_src=True)
    assert encoded_msg == [2] + [-1] * 511

    t = torch.rand(2, 512, 32, 32, 64)
    encoded_msg = ComputationModel._encode_input_data(t, is_src=True)
    dtype_str = str(t.dtype)
    true_msg = [0, 5, 2, 512, 32, 32, 64, len(dtype_str), *list(bytearray(dtype_str, "utf-8"))]
    assert encoded_msg == true_msg + [-1] * (512 - len(true_msg))

    t = torch.randint(-1235, 1233, size=(2, 512, 32, 32, 64))
    encoded_msg = ComputationModel._encode_input_data(t, is_src=True)
    dtype_str = str(t.dtype)
    true_msg = [0, 5, 2, 512, 32, 32, 64, len(dtype_str), *list(bytearray(dtype_str, "utf-8"))]
    assert encoded_msg == true_msg + [-1] * (512 - len(true_msg))

    t = torch.tensor(12)
    encoded_msg = ComputationModel._encode_input_data(t, is_src=True)
    dtype_str = str(t.dtype)
    true_msg = [0, 0, len(dtype_str), *list(bytearray(dtype_str, "utf-8"))]
    assert encoded_msg == true_msg + [-1] * (512 - len(true_msg))

    for t in [None, "abc", torch.rand(2, 512, 32, 32, 64), 12.34, object()]:
        encoded_msg = ComputationModel._encode_input_data(t, is_src=False)
        assert encoded_msg == [-1] * 512


def test__decode_as_placeholder():
    device = torch.device("cpu")

    encoded_msg = [-1] * 512
    encoded_msg[0] = 1
    res = ComputationModel._decode_as_placeholder(encoded_msg, device)
    assert isinstance(res, float) and res == 0.0

    encoded_msg = [-1] * 512
    encoded_msg[0] = 2
    res = ComputationModel._decode_as_placeholder(encoded_msg, device)
    assert isinstance(res, str) and res == ""

    encoded_msg = [-1] * 512
    encoded_msg[0] = 0
    encoded_msg[1 : 1 + 7] = [6, 2, 3, 4, 5, 6, 7]
    dtype_str = "torch.int64"
    payload = [len(dtype_str), *list(bytearray(dtype_str, "utf-8"))]
    encoded_msg[1 + 7 : 1 + 7 + len(payload)] = payload
    res = ComputationModel._decode_as_placeholder(encoded_msg, device)
    assert isinstance(res, torch.Tensor) and res.dtype == torch.int64 and res.shape == (2, 3, 4, 5, 6, 7)

    encoded_msg = [-1] * 512
    with pytest.raises(RuntimeError, match="Internal error: unhandled dtype"):
        ComputationModel._decode_as_placeholder(encoded_msg, device)

    t = torch.rand(2, 512, 32, 32, 64)
    encoded_msg = ComputationModel._encode_input_data(t, True)
    res = ComputationModel._decode_as_placeholder(encoded_msg, device)
    assert isinstance(res, torch.Tensor) and res.dtype == t.dtype and res.shape == t.shape

    t = torch.tensor(12)
    encoded_msg = ComputationModel._encode_input_data(t, True)
    res = ComputationModel._decode_as_placeholder(encoded_msg, device)
    assert isinstance(res, torch.Tensor) and res.dtype == t.dtype and res.shape == t.shape


def test__setup_placeholder():
    device = torch.device("cpu")

    from ignite.distributed.utils import _model

    for t in [torch.rand(2, 3, 4), "abc", 123.45]:
        data = _model._setup_placeholder(t, device, True)
        assert isinstance(data, type(t))
        if isinstance(data, torch.Tensor):
            assert (data == t).all()
        else:
            assert data == t
