from ignite.distributed.comp_models.base import _SerialModel
from ignite.distributed.comp_models.native import _NativeDistModel
from ignite.distributed.comp_models.xla import _XlaDistModel, has_xla_support

registered_computation_models = [_SerialModel, _NativeDistModel, _XlaDistModel]
