import itertools
import weakref
from typing import Optional

import torch
import torch.utils._pytree as pytree


def replace_node_with_constant(gm, node, constant):
    g = gm.graph

    i = 0
    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


def replace_params_with_constants(fake_gm, real_inputs, example_inputs_, fw_metadata):
    fake_inp_nodes = [node for (_, node) in zip(real_inputs, fake_gm.graph.nodes)]

    g = fake_gm.graph

    preserved_arg_indices = []

    for i, (real_input, fake_input, node) in enumerate(
        zip(real_inputs, example_inputs_, fake_inp_nodes)
    ):
        assert real_input.shape == fake_input.shape

        if i in fw_metadata.mutated_inp_indices:
            preserved_arg_indices.append(i)
            continue

        replace_node_with_constant(fake_gm, node, real_input)

    # add on non param inputs
    preserved_arg_indices.extend(range(len(real_inputs), len(example_inputs_)))

    g.lint()
    # is this necessary ?
    fake_gm.recompile()
    return fake_gm, preserved_arg_indices


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm, num_inputs):
    unknown_value = object()

    node_replacements = {}

    class ConstantFolder(torch.fx.Interpreter):
        def run_node(self, node):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            if unknown_value in pytree.tree_flatten((args, kwargs))[0]:
                return unknown_value

            if (
                isinstance(node.target, torch._ops.OpOverload)
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                return unknown_value

            # All mutations should either be removed or on inputs which we did not make constant
            # TODO - check rng state did not change ?

            out = super().run_node(node)

            if node.op != "get_attr" and isinstance(out, torch.Tensor):
                node_replacements[node] = out

            return out

    ConstantFolder(gm).run(*[unknown_value for _ in range(num_inputs)])

    for node, constant in node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def optimize_for_inference(
    original_gm: torch.fx.GraphModule,
    fake_gm: torch.fx.GraphModule,
    example_inputs_,
    fw_metadata,
):
    params = {
        **dict(original_gm.named_parameters(remove_duplicate=False)),
        **dict(original_gm.named_buffers(remove_duplicate=False)),
    }
    params_flat, _ = pytree.tree_flatten(params)
    params_flat = tuple(params_flat)

    fake_gm, preserved_arg_indices = replace_params_with_constants(
        fake_gm, params_flat, example_inputs_, fw_metadata
    )

    constant_fold(fake_gm, len(preserved_arg_indices))

    # invalidate nn Modules
    invalidate_eager_modules()
    return fake_gm, preserved_arg_indices


class ErasedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, name, owning_mod):
        return super().__new__(cls, elem.to(device="meta"))

    def __init__(self, elem, name: Optional[str], mod):
        self.erased_name = name
        self.owning_mod_ref = weakref.ref(mod)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise RuntimeError(
            f"Trying to Run Pytorch Eager Module After Dynamo Freezing. The original parameters have been discarded for memeory efficiency. "
            f"Found in op {func} for erased parameter {self.erased_name} of {self.owning_mod_ref()}"
        )


@torch.utils._python_dispatch._disable_current_modes()
def invalidate_eager_modules():
    # TODO - could just invalidate the parameters that were folded
    for mod in torch._guards.TracingContext.get().module_context.nn_modules.values():
        if not isinstance(mod, torch.nn.Module):
            continue

        for attr_name, tensor in list(
            itertools.chain(
                mod.named_parameters(recurse=False), mod.named_buffers(recurse=False)
            )
        ):
            e_t = ErasedTensor(tensor, attr_name, mod)
            if isinstance(tensor, torch.nn.Parameter):
                e_t.requires_grad_(True)
                e_t._is_param = True
            setattr(mod, attr_name, e_t)
