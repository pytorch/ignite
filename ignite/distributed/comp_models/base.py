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
        self._nproc_per_node = None
        self._nnodes = None
        self._node = None

    def _setup_attrs(self):
        if self._nproc_per_node is None:
            self._nproc_per_node = self._compute_nproc_per_node() if self.get_world_size() > 1 else 1
        if self._nnodes is None:
            self._nnodes = self.get_world_size() // self._nproc_per_node
        if self._node is None:
            self._node = self.get_rank() // self._nproc_per_node

    @abstractmethod
    def _compute_nproc_per_node(self) -> int:
        pass

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
    def get_nproc_per_node(self) -> int:
        pass

    @abstractmethod
    def get_nnodes(self) -> int:
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
        # output is tensor of shape (1, 1025)
        return padded_x.unsqueeze(0)

    @staticmethod
    def _decode_str(xs: torch.Tensor) -> List[str]:
        # xs.shape = (n, 1025), e.g. (world_size, 1025)
        out = [bytearray(x[: x[-1]].tolist()).decode("utf-8") for x in xs]
        return out

    def _apply_op(self, tensor: torch.Tensor, device: torch.device, fn: Callable, *args, **kwargs) -> torch.Tensor:
        out_dtype = None
        tensor_device = None

        # check if the tensor is at specified device
        if tensor.device != device:
            tensor_device = tensor.device
            tensor = tensor.to(device)

        if self._collective_op_dtype is not None:
            # cast to _collective_op_dtype if current type is not floatX
            if tensor.dtype not in (torch.float32, torch.float64):
                out_dtype = tensor.dtype
                tensor = tensor.to(self._collective_op_dtype)

        tensor = fn(tensor, *args, **kwargs)

        if out_dtype is not None and tensor_device is not None:
            return tensor.to(dtype=out_dtype, device=tensor_device)
        if out_dtype is not None:
            return tensor.to(dtype=out_dtype)
        if tensor_device is not None:
            return tensor.to(device=tensor_device)
        return tensor

    def _collective_op(
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

        tensor = self._apply_op(tensor, device, fn, *args, **kwargs)

        if tensor_to_number and tensor.numel() == 1:
            return tensor.item()
        elif tensor_to_str:
            return self._decode_str(tensor)
        return tensor

    def all_reduce(self, tensor: Union[torch.Tensor, Number], op: str = "sum") -> Union[torch.Tensor, Number]:
        if not isinstance(tensor, (torch.Tensor, Number)):
            raise TypeError("Unhandled input type {}".format(type(tensor)))

        return self._collective_op(tensor, self._do_all_reduce, op)

    def all_gather(self, tensor: Union[torch.Tensor, Number, str]) -> Union[torch.Tensor, Number, List[str]]:
        if not isinstance(tensor, (torch.Tensor, Number, str)):
            raise TypeError("Unhandled input type {}".format(type(tensor)))

        return self._collective_op(tensor, self._do_all_gather)

    def broadcast(self, tensor: Union[torch.Tensor, Number, str], src: int = 0) -> Union[torch.Tensor, Number, str]:
        if not isinstance(tensor, (torch.Tensor, Number, str)):
            raise TypeError("Unhandled input type {}".format(type(tensor)))

        rank = self.get_rank()
        device = self.device()
        tensor_to_number = tensor_to_str = False
        if rank != src:
            if isinstance(tensor, Number):
                tensor_to_number = True
                tensor = torch.empty(1, device=self.device(), dtype=torch.float)
            elif isinstance(tensor, str):
                tensor_to_str = True
                tensor = torch.empty(1, 1025, device=self.device(), dtype=torch.long)
        else:
            if isinstance(tensor, Number):
                tensor_to_number = True
                tensor = torch.tensor([tensor,], device=device, dtype=torch.float)
            elif isinstance(tensor, str):
                tensor_to_str = True
                tensor = self._encode_str(tensor, device)

        tensor = self._apply_op(tensor, device, self._do_broadcast, src)

        if tensor_to_number:
            return tensor.item()
        if tensor_to_str:
            list_str = self._decode_str(tensor)
            return list_str[0]
        return tensor

    @abstractmethod
    def _do_all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        pass

    @abstractmethod
    def _do_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        pass

    @abstractmethod
    def barrier(self):
        pass


class _SerialModel(ComputationModel):
    """Private class defines non-distributed computation model for code compatibility with other distributed models.
    """

    name = "serial"
    available_backends = ()

    def get_local_rank(self) -> int:
        return 0

    def get_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1

    def get_nproc_per_node(self) -> int:
        return 1

    def get_nnodes(self) -> int:
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

    def _compute_nproc_per_node(self) -> int:
        return 1

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

    def broadcast(self, tensor: Union[torch.Tensor, Number, str], src: int = 0) -> Union[torch.Tensor, Number, str]:
        return tensor

    def _do_all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        pass

    def _do_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        pass

    def barrier(self):
        pass
