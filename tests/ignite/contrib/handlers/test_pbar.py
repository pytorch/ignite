from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar


def test_with_engine(capsys):

    def update_fn(engine, batch):
        return {'a': 1, 'b': 2}

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)
    handler = ProgressBar(engine, loader)

    engine.add_event_handler(Events.ITERATION_COMPLETED, handler)
    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    out = captured.out.split('\n')
    out = list(filter(None, out))
    expected = 'Epoch {} | a={:.2e} | b={:.2e}'.format(engine.state.epoch, 1, 2)
    assert len(out) == n_epochs
    assert out[-1] == expected
