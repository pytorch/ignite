from typing import List, Tuple, Type, TYPE_CHECKING, Union

from ignite.distributed.comp_models.base import _SerialModel
from ignite.distributed.comp_models.horovod import has_hvd_support
from ignite.distributed.comp_models.native import has_native_dist_support
from ignite.distributed.comp_models.xla import has_xla_support

if TYPE_CHECKING:
    from ignite.distributed.comp_models.horovod import _HorovodDistModel
    from ignite.distributed.comp_models.native import _NativeDistModel
    from ignite.distributed.comp_models.xla import _XlaDistModel


def setup_available_computation_models() -> Tuple[
    Type[Union[_SerialModel, "_NativeDistModel", "_XlaDistModel", "_HorovodDistModel"]], ...
]:
    models: List[Type[Union[_SerialModel, "_NativeDistModel", "_XlaDistModel", "_HorovodDistModel"]]] = [
        _SerialModel,
    ]
    if has_native_dist_support:
        from ignite.distributed.comp_models.native import _NativeDistModel

        models.append(_NativeDistModel)
    if has_xla_support:
        from ignite.distributed.comp_models.xla import _XlaDistModel

        models.append(_XlaDistModel)
    if has_hvd_support:
        from ignite.distributed.comp_models.horovod import _HorovodDistModel

        models.append(_HorovodDistModel)

    return tuple(models)


registered_computation_models = setup_available_computation_models()
