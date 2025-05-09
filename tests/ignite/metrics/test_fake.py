import pytest
import torch

from ignite.metrics import Fbeta, Precision, Recall

torch.manual_seed(12)


# FIXME: the file needs to be removed
def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"Beta should be a positive integer"):
        Fbeta(0.0)

    with pytest.raises(ValueError, match=r"Input precision metric should have average=False"):
        p = Precision(average="micro")
        Fbeta(1.0, precision=p)

    with pytest.raises(ValueError, match=r"Input recall metric should have average=False"):
        r = Recall(average="samples")
        Fbeta(1.0, recall=r)

    with pytest.raises(ValueError, match=r"If precision argument is provided, device should be None"):
        p = Precision(average=False)
        Fbeta(1.0, precision=p, device="cpu")

    with pytest.raises(ValueError, match=r"If precision argument is provided, output_transform should be None"):
        p = Precision(average=False)
        Fbeta(1.0, precision=p, output_transform=lambda x: x)

    with pytest.raises(ValueError, match=r"If recall argument is provided, device should be None"):
        r = Recall(average=False)
        Fbeta(1.0, recall=r, device="cpu")

    with pytest.raises(ValueError, match=r"If recall argument is provided, output_transform should be None"):
        r = Recall(average=False)
        Fbeta(1.0, recall=r, output_transform=lambda x: x)


def _output_transform(output):
    return output["y_pred"], output["y"]
