# coding: utf-8

from enum import Enum

import ignite.engine.engine


def add_events(Events):
    """Add events to core ignite.

    Add envents to `ignite.engine` and `ignite.engine.engine`.

    Args:
        Events (Enum): A enum with new supported events.
    """
    prev_events = tuple(ignite.engine.Events.__members__.keys())
    new_events = tuple(Events.__members__.keys())
    NewEnum = Enum(
        ignite.engine.engine.Events.__name__, prev_events + new_events
    )
    ignite.engine.engine.Events = NewEnum
    ignite.engine.Events = NewEnum
