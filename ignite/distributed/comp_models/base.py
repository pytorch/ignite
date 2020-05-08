from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import torch


class ComputationModel(metaclass=ABCMeta):

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
    def is_distributed(self) -> bool:
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


class _SerialModel(ComputationModel):

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
