import warnings
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Callable, List, Optional, Union

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
    def device(self) -> torch.device:
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

    _collective_op_dtype = None

    @staticmethod
    def _encode_str(x: str, device: torch.device) -> torch.Tensor:
        # use fix padded size
        size = 1024
        if len(x) > size:
            warnings.warn("Input string size {} is larger than {} and thus will be truncated".format(len(x), size))
            x = x[:size]

        name = torch.tensor(bytearray(x, "utf-8")).to(device)
        padded_x = torch.zeros(size + 1, device=device, dtype=torch.long)
        padded_x[: len(name)] = name
        padded_x[-1] = len(name)
        return padded_x.unsqueeze(0)

    @staticmethod
    def _decode_str(xs: torch.Tensor) -> List[str]:
        # xs.shape = (world_size, 1025)
        out = [bytearray(x[: x[-1]].tolist()).decode("utf-8") for x in xs]
        return out

    def _all_collective_op(
        self, tensor: Union[torch.Tensor, Number, str], fn: Callable, *args, **kwargs
    ) -> Union[torch.Tensor, Number, List[str]]:
        tensor_to_number = tensor_to_str = False
        device = self.device()
        if isinstance(tensor, Number):
            tensor_to_number = True
            tensor = torch.tensor(tensor, device=device, dtype=self._collective_op_dtype)
        elif isinstance(tensor, str):
            tensor_to_str = True
            tensor = self._encode_str(tensor, device)

        out_dtype = None

        # check if the tensor is at specified device
        if tensor.device != device:
            tensor = tensor.to(device)
        if self._collective_op_dtype is not None:
            # cast to _collective_op_dtype if current type is not floatX
            if tensor.dtype not in (torch.float32, torch.float64):
                out_dtype = tensor.dtype
                tensor = tensor.to(self._collective_op_dtype)

        tensor = fn(tensor, *args, **kwargs)

        if out_dtype is not None:
            tensor = tensor.to(dtype=out_dtype)

        if tensor_to_number and tensor.numel() == 1:
            return tensor.item()
        elif tensor_to_str:
            return self._decode_str(tensor)
        return tensor

    def all_reduce(self, tensor: Union[torch.Tensor, Number], op: str = "sum") -> Union[torch.Tensor, Number]:
        if not isinstance(tensor, (torch.Tensor, Number)):
            raise TypeError("Unhandled input type {}".format(type(tensor)))

        return self._all_collective_op(tensor, self._do_all_reduce, op)

    def all_gather(self, tensor: Union[torch.Tensor, Number, str]) -> Union[torch.Tensor, Number, List[str]]:
        if not isinstance(tensor, (torch.Tensor, Number, str)):
            raise TypeError("Unhandled input type {}".format(type(tensor)))

        return self._all_collective_op(tensor, self._do_all_gather)

    @abstractmethod
    def _do_all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        pass

    @abstractmethod
    def _do_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def barrier(self):
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

    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def backend(self) -> None:
        return None

    def finalize(self):
        pass

    @staticmethod
    def create_from_context() -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def create_from_backend(backend: Optional[str] = None, **kwargs) -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def spawn(*args, **kwargs):
        raise NotImplementedError("Serial computation model does not implement spawn method")

    def all_reduce(self, tensor: Union[torch.Tensor, Number], op: str = "sum") -> Union[torch.Tensor, Number]:
        return tensor

    def all_gather(self, tensor: Union[torch.Tensor, Number]) -> Union[torch.Tensor, Number]:
        return tensor

    def _do_all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        pass

    def _do_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    def barrier(self):
        pass
