from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Any, Callable, cast, List, Optional, Union

import torch


class ComputationModel(metaclass=ABCMeta):
    """Base class for distributed computation models and defines interface methods.
    This class is public and should be used for other custom derived distributed models.
    """

    # this is an additional local rank storage used when idist is setup from existing native torch dist context
    _ext_local_rank = None  # type: Optional[int]

    def __init__(self) -> None:
        self._backend = None  # type: Optional[str]
        self._nproc_per_node = None  # type: Optional[int]
        self._nnodes = None  # type: Optional[int]
        self._node = None  # type: Optional[int]

    def _setup_attrs(self) -> None:
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
    def finalize(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def create_from_context() -> Optional["ComputationModel"]:
        pass

    @staticmethod
    @abstractmethod
    def create_from_backend(backend: str, **kwargs: Any) -> "ComputationModel":
        pass

    @staticmethod
    @abstractmethod
    def spawn(*args: Any, **kwargs: Any) -> None:
        pass

    _collective_op_dtype = None  # type: Any

    @staticmethod
    def _encode_str(x: str, device: torch.device, size: int) -> torch.Tensor:
        name = torch.tensor(bytearray(x, "utf-8")).to(device)
        padded_x = torch.zeros(size + 1, device=device, dtype=torch.long)
        padded_x[: len(name)] = name
        padded_x[-1] = len(name)
        # output is tensor of shape (1, size + 1)
        return padded_x.unsqueeze(0)

    def _get_max_length(self, x: str, device: torch.device) -> int:
        size = torch.tensor([len(x)], device=device)
        size = self._do_all_reduce(size, op="MAX")
        return cast(int, size.item())

    @staticmethod
    def _encode_input_data(x: Union[torch.Tensor, float, str, None], is_src: bool) -> List[int]:
        encoded_msg = [-1] * 512
        if not is_src:
            # Discard input type if not source
            return encoded_msg

        if isinstance(x, torch.Tensor):
            shape = x.shape
            dtype = str(x.dtype)
            msg = [0, len(shape), *shape, len(dtype), *list(bytearray(dtype, "utf-8"))]
            encoded_msg[: len(msg)] = msg
        elif isinstance(x, Number):
            encoded_msg[0] = 1
        elif isinstance(x, str):
            encoded_msg[0] = 2
        return encoded_msg

    @staticmethod
    def _decode_as_placeholder(encoded_msg: List[int], device: torch.device) -> Union[torch.Tensor, float, str]:
        if encoded_msg[0] == 0:
            len_shape = encoded_msg[1]
            le = 2 + len_shape
            shape = encoded_msg[2:le] if len_shape > 0 else []
            len_dtype = encoded_msg[le]
            dtype_str = bytearray(encoded_msg[le + 1 : le + 1 + len_dtype]).decode("utf-8")
            dtype = eval(dtype_str)
            return torch.empty(shape, device=device, dtype=dtype)
        elif encoded_msg[0] == 1:
            return 0.0
        elif encoded_msg[0] == 2:
            return ""
        else:
            raise RuntimeError(f"Internal error: unhandled dtype {encoded_msg[0]}, encoded_msg={encoded_msg}")

    def _setup_placeholder(
        self, x: Union[torch.Tensor, float, str, None], device: torch.device, is_src: bool
    ) -> Union[torch.Tensor, float, str]:

        encoded_msg_per_rank = self._encode_input_data(x, is_src)
        encoded_msg_all_ranks = self._do_all_reduce(torch.tensor(encoded_msg_per_rank, device=device), op="MAX")

        if is_src:
            if x is None:
                raise RuntimeError("Internal error, x is None. Please, file an issue if you encounter this error.")
            return x

        encoded_msg = encoded_msg_all_ranks.cpu().tolist()
        return self._decode_as_placeholder(encoded_msg, device)

    @staticmethod
    def _decode_str(xs: torch.Tensor) -> List[str]:
        # xs.shape = (n, size + 1), e.g. (world_size, size + 1)
        out = [bytearray(x[: x[-1]].tolist()).decode("utf-8") for x in xs]
        return out

    def _apply_op(
        self, tensor: torch.Tensor, device: torch.device, fn: Callable, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
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
        self, tensor: Union[torch.Tensor, float, str], fn: Callable, *args: Any, **kwargs: Any
    ) -> Union[torch.Tensor, float, List[float], List[str]]:
        tensor_to_number = tensor_to_str = False
        device = self.device()
        if isinstance(tensor, (Number, float)):
            tensor_to_number = True
            tensor = torch.tensor(tensor, device=device, dtype=self._collective_op_dtype)
        elif isinstance(tensor, str):
            tensor_to_str = True
            max_length = self._get_max_length(tensor, device)
            tensor = self._encode_str(tensor, device, size=max_length)

        tensor = self._apply_op(tensor, device, fn, *args, **kwargs)

        if tensor_to_number:
            if tensor.numel() == 1:
                return tensor.item()
            else:
                return tensor.tolist()
        elif tensor_to_str:
            return self._decode_str(tensor)
        return tensor

    def all_reduce(
        self, tensor: Union[torch.Tensor, float], op: str = "sum", group: Optional[Any] = None
    ) -> Union[torch.Tensor, float]:
        if not isinstance(tensor, (torch.Tensor, Number)):
            raise TypeError(f"Unhandled input type {type(tensor)}")

        return cast(Union[torch.Tensor, float], self._collective_op(tensor, self._do_all_reduce, op, group=group))

    def all_gather(
        self, tensor: Union[torch.Tensor, float, str], group: Optional[Any] = None
    ) -> Union[torch.Tensor, float, List[float], List[str]]:
        if not isinstance(tensor, (torch.Tensor, Number, str)):
            raise TypeError(f"Unhandled input type {type(tensor)}")

        return self._collective_op(tensor, self._do_all_gather, group=group)

    def new_group(self, ranks: List[int], **kwargs: Any) -> Any:
        if isinstance(ranks, list) and all(isinstance(item, int) for item in ranks):
            return self._do_new_group(ranks, **kwargs)
        else:
            raise ValueError("Argument ranks should be list of int")

    def broadcast(
        self, tensor: Union[torch.Tensor, float, str, None], src: int = 0, safe_mode: bool = False
    ) -> Union[torch.Tensor, float, str]:
        if not (isinstance(tensor, (torch.Tensor, Number, str)) or tensor is None):
            raise TypeError(f"Unhandled input type {type(tensor)}")

        rank = self.get_rank()
        if tensor is None:
            if rank == src:
                raise ValueError("Source data can not be None")
            elif not safe_mode:
                raise ValueError("Argument safe_mode should be True if None is passed for non src rank")

        device = self.device()
        tensor_to_number = tensor_to_str = False

        if safe_mode:
            tensor = self._setup_placeholder(tensor, device, rank == src)

        if tensor is None:
            raise RuntimeError("Internal error, tensor is None. Please, file an issue if you encounter this error.")

        if isinstance(tensor, (Number, float)):  # have to use Number and float to satisfy mypy
            tensor_to_number = True
            if rank != src:
                tensor = torch.empty(1, device=device, dtype=torch.float)
            else:
                tensor = torch.tensor([tensor], device=device, dtype=torch.float)
        elif isinstance(tensor, str):
            tensor_to_str = True
            max_length = self._get_max_length(tensor, device)
            if rank != src:
                tensor = torch.empty(1, max_length + 1, device=device, dtype=torch.long)
            else:
                tensor = self._encode_str(tensor, device, size=max_length)

        tensor = self._apply_op(tensor, device, self._do_broadcast, src)

        if tensor_to_number:
            return tensor.item()
        if tensor_to_str:
            list_str = self._decode_str(tensor)
            return list_str[0]
        return tensor

    @abstractmethod
    def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM", group: Optional[Any] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def _do_all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        pass

    @abstractmethod
    def barrier(self) -> None:
        pass

    @abstractmethod
    def _do_new_group(self, ranks: List[int], **kwargs: Any) -> Any:
        pass


class _SerialModel(ComputationModel):
    """Private class defines non-distributed computation model for code compatibility with other distributed models."""

    name = "serial"
    available_backends = ()

    def __init__(self, _backend: Optional[str] = None, **kwargs: Any) -> None:
        super(_SerialModel, self).__init__()

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

    def backend(self) -> Optional[str]:
        return None

    def finalize(self) -> None:
        pass

    def _compute_nproc_per_node(self) -> int:
        return 1

    @staticmethod
    def create_from_context() -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def create_from_backend(backend: Optional[str] = None, **kwargs: Any) -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def spawn(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Serial computation model does not implement spawn method")

    def all_reduce(
        self, tensor: Union[torch.Tensor, float], op: str = "SUM", group: Optional[Any] = None
    ) -> Union[torch.Tensor, float]:
        return tensor

    def all_gather(
        self, tensor: Union[torch.Tensor, float, str], group: Optional[Any] = None
    ) -> Union[torch.Tensor, float, List[float], List[str]]:
        if isinstance(tensor, torch.Tensor):
            return tensor
        return cast(Union[List[float], List[str]], [tensor])

    def broadcast(
        self, tensor: Union[torch.Tensor, float, str, None], src: int = 0, safe_mode: bool = False
    ) -> Union[torch.Tensor, float, str]:
        if tensor is None:
            raise ValueError("Argument tensor should not be None")
        return tensor

    def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM", group: Optional[Any] = None) -> torch.Tensor:
        return tensor

    def _do_all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None) -> torch.Tensor:
        return tensor

    def _do_new_group(self, ranks: List[int], **kwargs: Any) -> Any:
        return ranks

    def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        return tensor

    def barrier(self) -> None:
        pass

    def new_group(self, ranks: List[int], **kwargs: Any) -> Any:
        if isinstance(ranks, list) and all(isinstance(item, int) for item in ranks):
            return self._do_new_group(ranks, **kwargs)
        else:
            raise ValueError("Argument ranks should be list of int")
