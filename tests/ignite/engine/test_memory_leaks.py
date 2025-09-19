import weakref

import pytest

from ignite.engine import Engine, Events


class TestEngineMemoryLeak:
    """See: https://github.com/pytorch/ignite/issues/3438"""

    ENGINE_WEAK_REFS = set()

    def do_train(self, cls, with_handler) -> None:
        engine = cls(lambda e, b: None)

        if with_handler:

            @engine.on(Events.EPOCH_STARTED)
            def handler(engine) -> None:
                pass

        engine.run(range(5), max_epochs=5)
        self.ENGINE_WEAK_REFS.add(weakref.ref(engine))

    @pytest.mark.parametrize("with_handler", [True, False])
    def test_memory_leak(self, with_handler):
        num_iters = 5
        counter = 0

        class EngineForTests(Engine):

            def __del__(self):
                nonlocal counter
                counter += 1

        for i in range(num_iters):
            self.do_train(EngineForTests, with_handler)
            for weak_engine_ref in self.ENGINE_WEAK_REFS:
                engine = weak_engine_ref()
                assert engine is None

            assert counter == i + 1

        assert counter == i + 1
