from ignite.engine import Engine
from ignite.metrics import Loss
from ignite.contrib.handlers import ProgressBar


def update_fn(engine, batch):
    engine.state.metrics['a'] = 1
    engine.state.metrics['b'] = 2
    return None


def test_epoch_mode(capsys):

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine, len(loader), ['a'])
    pbar.add_logging(engine, ['a', 'b'], mode='epoch', log_interval=1)

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    out = captured.out.split('\n')
    out = list(filter(None, out))
    expected = 'Epoch {} | a={:.2e} | b={:.2e}'.format(engine.state.epoch, 1, 2)
    expected2 = 'Epoch {} | b={:.2e} | a={:.2e}'.format(engine.state.epoch, 2, 1)
    assert len(out) == n_epochs
    assert out[-1] == expected or out[-1] == expected2


def test_iteration_mode(capsys):

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine, len(loader), ['a'])
    pbar.add_logging(engine, ['a', 'b'], mode='iteration', log_interval=1)

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    out = captured.out.split('\n')
    out = list(filter(None, out))
    expected = 'Iteration {} | a={:.2e} | b={:.2e}'.format(engine.state.iteration, 1, 2)
    expected2 = 'Iteration {} | b={:.2e} | a={:.2e}'.format(engine.state.iteration, 2, 1)
    assert len(out) == n_epochs * len(loader)
    assert out[-1] == expected or out[-1] == expected2
