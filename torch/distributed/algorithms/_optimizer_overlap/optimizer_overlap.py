from abc import ABC
import inspect
from typing import Dict, Type

from torch.distributed._fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.distributed.optim import as_functional_optim

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
    _OptimizerHookState,
    _hook_then_optimizer
)

# Contains the mappings between the regular and overlapped optimizer types.
_registered_overlapped_optims: Dict[Type, Type] = {}


def register_overlapped(optim_cls):
    def decorator(target_overlapped_optim_cls):
        if target_overlapped_optim_cls in _registered_overlapped_optims:
            raise ValueError(
                f"{target_overlapped_optim_cls} already registered with optim_cls "
                f"{_registered_overlapped_optims[optim_cls]} {optim_cls}, trying to"
                f"re-register it for {optim_cls} is not supported."
            )
        _registered_overlapped_optims[optim_cls] = target_overlapped_optim_cls
        return target_overlapped_optim_cls
    return decorator


class OverlappedOptimizer(ABC):
    def __init__(self, optim_cls: Type) -> None:
        """
        OverlappedOptimizer is a base class that child classes can implement to
        specify how different optimizers will register themselves with DDP.
        """
        self.optim_cls = optim_cls

    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        """Registers the overlapped optimizer with DDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped DDP."
        )

    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Registers the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )


@register_overlapped(Optimizer)
class _OverlappedStandardOptimizer(OverlappedOptimizer):
    """Overlaps a regular ``Optimizer``."""

    def __init__(self, optim_cls: Type, params, *optim_args, **optim_kwargs) -> None:
        super().__init__(optim_cls)
        f_optim = as_functional_optim(self.optim_cls, *optim_args, **optim_kwargs)
        self._opt_hook_state = _OptimizerHookState.from_functional_optim(f_optim, params)

    def register_ddp(self, ddp_inst: DistributedDataParallel):
        # NOTE: using a custom communication hook and fused optimizer is not
        # yet supported.
        ddp_inst.register_comm_hook(  # type: ignore[operator]
            None,  # wrapped hook state
            _hook_then_optimizer(allreduce_hook, self._opt_hook_state)
        )

    # TODO: register_fsdp once FSDP supports communication hook.


def _as_overlapped_optim(optim_cls: Type, params, *args, **kwargs):
    """
    Returns a new ``OverlappedOptimizer`` instance that supports ``optim_cls``.
    """
    for clz in inspect.getmro(optim_cls):
        try:
            return _registered_overlapped_optims[clz](optim_cls, params, *args, **kwargs)
        except KeyError:
            pass

    # Fallback to standard overlapped optimizer, which will raise errors if user
    # is attempting to use an unsupported optimizer.
    return _OverlappedStandardOptimizer(optim_cls, params, *args, **kwargs)
