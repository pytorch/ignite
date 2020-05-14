from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Optional, Union

import torch


class ComputationModel(metaclass=ABCMeta):
    """Base class for distributed computation models and defines interface methods.

    This class is public and should be used for other custom derived distributed models.
    """

    # this is an additional local rank storage used when idist is setup from existing native torch dist context
    _ext_local_rank = None

    def __init__(self):
        self._backend = None
        self._ntasks_per_node = None
        self._nnodes = None
        self._node = None

    @abstractmethod
    def get_local_rank(self) -> int:
        pass

    @abstractmethod
    def get_rank(self) -> int:
        pass

    @abstractmethod
    def get_world_size(self) -> int:
        pass

    @abstractmethod
    def get_ntasks_per_node(self) -> int:
        pass

    @abstractmethod
    def get_num_nodes(self) -> int:
        pass

    @abstractmethod
    def get_node_rank(self) -> int:
        pass

    @abstractmethod
    def device(self) -> Union[torch.device, str]:
        pass

    @abstractmethod
    def backend(self) -> Optional[str]:
        pass

    @abstractmethod
    def finalize(self):
        pass

    @staticmethod
    @abstractmethod
    def create_from_context() -> Optional["ComputationModel"]:
        pass

    @staticmethod
    @abstractmethod
    def create_from_backend(backend: str, **kwargs) -> "ComputationModel":
        pass

    @staticmethod
    @abstractmethod
    def spawn(*args, **kwargs):
        pass

    _reduction_dtype = None

    def all_reduce(self, tensor: Union[torch.Tensor, Number], op: str = "sum") -> Union[torch.Tensor, Number]:
        tensor_to_number = False
        device = self.device()
        if isinstance(tensor, Number):
            tensor = torch.tensor(tensor, device=device, dtype=self._reduction_dtype)
            tensor_to_number = True

        if isinstance(tensor, torch.Tensor):
            # check if the tensor is at specified device
            if tensor.device != device:
                tensor = tensor.to(device)
        else:
            raise TypeError("Unhandled input type {}".format(type(tensor)))

        self._do_reduction(tensor, op)

        if tensor_to_number:
            return tensor.item()
        return tensor

    @abstractmethod
    def _do_reduction(self, tensor: Union[torch.Tensor, Number], op: str = "sum"):
        pass


class _SerialModel(ComputationModel):
    """Private class defines non-distributed computation model for code compatibility with other distributed models.
    """

    name = "serial"
    available_backends = tuple()

    def get_local_rank(self) -> int:
        return 0

    def get_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1

    def get_ntasks_per_node(self) -> int:
        return 1

    def get_num_nodes(self) -> int:
        return 1

    def get_node_rank(self) -> int:
        return 0

    def is_distributed(self) -> bool:
        return False

    def device(self) -> Union[torch.device, str]:
        return "cpu"

    def backend(self) -> Optional[str]:
        return None

    def finalize(self):
        pass

    @staticmethod
    def create_from_context() -> Optional["_SerialModel"]:
        return _SerialModel()

    @staticmethod
    def create_from_backend(backend: str, **kwargs) -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def spawn(*args, **kwargs):
        raise NotImplementedError("Serial computation model does not implement spawn method")

    def all_reduce(self, tensor: Union[torch.Tensor, Number], op: str = "sum") -> Union[torch.Tensor, Number]:
        return tensor

    def _do_reduction(self, tensor: Union[torch.Tensor, Number], op: str = "sum"):
        pass
