from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar


def update_fn(engine, batch):
    engine.state.metrics['a'] = 1
    return None


def test_pbar(capsys):

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine, ['a'])

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = 'Epoch 2: [1/2]  50%|█████     , a=1.00e+00 [00:00<00:00]'
    assert err[-1] == expected
