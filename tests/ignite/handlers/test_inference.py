from pytest import raises

from ignite.handlers import Inference


def test_inference_init():
    with raises(ValueError):
        Inference([], iteration_interval=1, epoch_interval=1)

    with raises(ValueError):
        Inference([])
