"""Apply a translated AsymFLUX.2 adapter to a stock ComfyUI FLUX.2 transformer.

Two halves:
  1. Direct, structural surgery on the inner `Flux` nn.Module (replace img_in
     and final_layer.linear; register proj_buffer/scale_buffer). These are
     tiny and don't require the model's weights to be on CPU.
  2. A patches dict that's handed to ComfyUI's ModelPatcher.add_patches.
     ComfyUI applies these LAZILY at weight-load time, so we never have to
     yank the 18 GB base model off the GPU.

For the LoRA half we precompute the dense delta `B @ A * scaling` in fp32
(one at a time, ~50-300 MB peak), store it back in fp16, and submit it as a
`("diff", (delta,))` patch. That keeps everything inside ComfyUI's existing
weight-management machinery.
"""

import torch
import torch.nn as nn

try:
    from .key_map import (
        translate, LORA_SCALING,
        ASYMFLUX_HIDDEN, ASYMFLUX_IN_CHANNELS,
        ASYMFLUX_OUT_CHANNELS, ASYMFLUX_PATCH_SIZE,
    )
except ImportError:
    from key_map import (
        translate, LORA_SCALING,
        ASYMFLUX_HIDDEN, ASYMFLUX_IN_CHANNELS,
        ASYMFLUX_OUT_CHANNELS, ASYMFLUX_PATCH_SIZE,
    )


def _get_submodule(root: nn.Module, dotted_path: str) -> nn.Module:
    obj = root
    for part in dotted_path.split('.'):
        obj = getattr(obj, part)
    return obj


def _set_submodule(root: nn.Module, dotted_path: str, value: nn.Module) -> None:
    parts = dotted_path.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _get_param(root: nn.Module, dotted_path: str) -> torch.nn.Parameter:
    *module_parts, leaf = dotted_path.split('.')
    mod = root
    for part in module_parts:
        mod = getattr(mod, part)
    return getattr(mod, leaf)


def _fuse_lora_into_weight(weight: torch.nn.Parameter,
                           lora_A: torch.Tensor,
                           lora_B: torch.Tensor,
                           scaling: float) -> None:
    """In-place: weight += (B @ A) * scaling, computed in float32. Kept for the
    unit tests; the runtime loader uses ComfyUI patches instead."""
    out_features, in_features = weight.shape
    assert lora_A.shape == (lora_B.shape[1], in_features)
    assert lora_B.shape == (out_features, lora_A.shape[0])
    delta = (lora_B.to(torch.float32, copy=False)
             @ lora_A.to(torch.float32, copy=False)) * scaling
    with torch.no_grad():
        weight.add_(delta.to(dtype=weight.dtype, device=weight.device))


def _materialize_lora_delta(A: torch.Tensor, B: torch.Tensor,
                            scaling: float,
                            store_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """delta = B @ A * scaling. Compute in fp32 on CPU, store in `store_dtype`."""
    # Always keep A and B on CPU during this -- we don't want to spuriously
    # move them through GPU just to do a matmul.
    A_cpu = A.detach().to(device='cpu', dtype=torch.float32, copy=False)
    B_cpu = B.detach().to(device='cpu', dtype=torch.float32, copy=False)
    return (B_cpu @ A_cpu * scaling).to(dtype=store_dtype)


# ---------------------------------------------------------------------------
# Surgery + patch building
# ---------------------------------------------------------------------------

def apply_structural_surgery(transformer: nn.Module,
                             plan,
                             compute_dtype: torch.dtype) -> dict:
    """Replace img_in / final_layer.linear; register proj_buffer / scale_buffer.

    These are tiny operations (a few MB total) that mutate the nn.Module
    directly. The base model's existing weights are untouched on whatever
    device they currently live on.
    """
    manifest = {'replaced_modules': [], 'registered_buffers': [], 'skipped': []}

    # Pick a device that exists in the model. If the model has no params yet
    # (shouldn't happen but be safe), default to CPU.
    try:
        ref_device = next(transformer.parameters()).device
    except StopIteration:
        ref_device = torch.device('cpu')

    for bfl_path, (expected_shape, tensor) in plan.module_replacements.items():
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f'{bfl_path}: adapter tensor shape {tuple(tensor.shape)} != '
                f'expected {expected_shape}')
        out_features, in_features = expected_shape
        try:
            existing = _get_submodule(transformer, bfl_path)
        except AttributeError:
            raise AttributeError(f'transformer has no submodule "{bfl_path}"')
        existing_device = next(existing.parameters()).device if any(True for _ in existing.parameters()) else ref_device
        new_linear = nn.Linear(in_features, out_features, bias=False,
                               device=existing_device, dtype=compute_dtype)
        with torch.no_grad():
            new_linear.weight.copy_(tensor.to(dtype=compute_dtype, device=existing_device))
        _set_submodule(transformer, bfl_path, new_linear)
        manifest['replaced_modules'].append(bfl_path)

    for name, tensor in plan.buffers.items():
        transformer.register_buffer(
            name, tensor.to(device=ref_device).clone(), persistent=False)
        manifest['registered_buffers'].append(name)

    return manifest


def build_comfy_patches(plan, key_prefix: str = 'diffusion_model.') -> dict:
    """Translate weight overrides + LoRA pairs into a ComfyUI patches dict.

    Output:
        {
            f'{key_prefix}final_layer.adaLN_modulation.1.weight':
                ('set', (tensor_fp16,)),
            f'{key_prefix}double_blocks.0.img_mlp.0.weight':
                ('diff', (delta_fp16,)),
            ...
        }

    `("set", ...)` replaces the weight tensor entirely at load time.
    `("diff", ...)` adds the delta tensor to the weight at load time.

    Both are handled by `comfy.lora.calculate_weight` -- no custom dispatch needed.
    """
    patches = {}

    # Full-weight overrides -> ("set", ...)
    for bfl_param_path, tensor in plan.weight_overrides.items():
        patches[f'{key_prefix}{bfl_param_path}'] = (
            'set', (tensor.to(torch.float16),))

    # LoRA pairs -> precompute dense delta, ("diff", ...).
    for bfl_module_path, halves in plan.lora_pairs.items():
        if set(halves.keys()) != {'A', 'B'}:
            raise KeyError(f'{bfl_module_path}: incomplete LoRA pair {list(halves)}')
        delta = _materialize_lora_delta(halves['A'], halves['B'], LORA_SCALING)
        patches[f'{key_prefix}{bfl_module_path}.weight'] = ('diff', (delta,))

    return patches


# ---------------------------------------------------------------------------
# Legacy single-shot API (kept for unit tests; not used by the live loader)
# ---------------------------------------------------------------------------

def apply_adapter(transformer: nn.Module,
                  adapter_state_dict: dict,
                  compute_dtype: torch.dtype = torch.bfloat16,
                  strict: bool = True) -> dict:
    """In-place full-fusion. Kept ONLY for the synthetic unit tests, where we
    can't pipe through ComfyUI's ModelPatcher. The production loader in
    nodes.py uses apply_structural_surgery + build_comfy_patches instead."""
    plan = translate(adapter_state_dict)

    if plan.unknown_keys:
        msg = f'Unknown adapter keys ({len(plan.unknown_keys)}): {plan.unknown_keys[:5]}...'
        if strict:
            raise KeyError(msg)
        print(f'[asymflux2] warning: {msg}')

    manifest = apply_structural_surgery(transformer, plan, compute_dtype)
    manifest.update({'overwrote_weights': [], 'fused_loras': []})

    for bfl_param_path, tensor in plan.weight_overrides.items():
        try:
            param = _get_param(transformer, bfl_param_path)
        except AttributeError:
            if strict:
                raise
            manifest['skipped'].append(bfl_param_path)
            continue
        if tuple(param.shape) != tuple(tensor.shape):
            raise ValueError(f'{bfl_param_path}: base shape {tuple(param.shape)} != '
                             f'adapter shape {tuple(tensor.shape)}')
        with torch.no_grad():
            param.copy_(tensor.to(dtype=param.dtype, device=param.device))
        manifest['overwrote_weights'].append(bfl_param_path)

    for bfl_module_path, halves in plan.lora_pairs.items():
        if set(halves.keys()) != {'A', 'B'}:
            if strict:
                raise KeyError(f'{bfl_module_path}: incomplete LoRA pair')
            manifest['skipped'].append(bfl_module_path)
            continue
        try:
            module = _get_submodule(transformer, bfl_module_path)
        except AttributeError:
            if strict:
                raise
            manifest['skipped'].append(bfl_module_path)
            continue
        _fuse_lora_into_weight(module.weight, halves['A'], halves['B'], LORA_SCALING)
        manifest['fused_loras'].append(bfl_module_path)

    return manifest


def asymflux2_model_config() -> dict:
    return {
        'patch_size':   ASYMFLUX_PATCH_SIZE,
        'in_channels':  ASYMFLUX_IN_CHANNELS,
        'out_channels': ASYMFLUX_OUT_CHANNELS,
        'hidden_size':  ASYMFLUX_HIDDEN,
    }
