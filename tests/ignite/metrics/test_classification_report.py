import pytest
import torch

from ignite.metrics.classification_report import ClassificationReport

torch.manual_seed(12)


@pytest.mark.parametrize("output_dict", [True])
def test_binary_input_N(output_dict):

    classification_report = ClassificationReport(output_dict=output_dict)

    def _test(y_true, y_pred, n_iters):

        y_true = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]])
        classification_report.reset()

        if n_iters > 1:
            batch_size = y_true.shape[0] // n_iters + 1
            for i in range(n_iters):
                idx = i * batch_size
                classification_report.update((y_true[idx : idx + batch_size], y_pred[idx : idx + batch_size]))
        else:
            classification_report.update((y_true, y_pred))

        from sklearn.metrics import classification_report as sklearn_classification_report

        y_true_transformed = [element.cpu().numpy().tolist().index(1) for element in y_true]

        res = classification_report.compute()
        assert res == sklearn_classification_report(y_true_transformed, y_pred, output_dict=True)
        assert isinstance(res, dict if output_dict else str)

    def get_test_cases():
        test_cases = [
            (torch.randint(0, 2, size=(10, 2)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
        ]
        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_true, y_pred, n_iters in test_cases:
            _test(y_true, y_pred, n_iters)
