"""ComfyUI nodes for Lakonik's AsymFLUX.2-klein adapter.

Workflow:

    [UNETLoader (flux2-klein-base-9b)] -> MODEL
              \\
               -> [AsymFlux2LoadAdapter (path=...)] -> MODEL  (patched)
                              |
    [CLIP encode +/-] -> COND
    [AsymFlux2EmptyLatent (w,h)] -> LATENT  (3-channel pixel-space noise)
    [AsymFlux2Sigmas (steps,w,h)] -> SIGMAS
    [KSamplerSelect (uni_pc)] -> SAMPLER
              \\
               -> [SamplerCustom] -> LATENT  (denoised, 3-channel)
                              |
                              -> [OklabDecode] -> IMAGE -> [SaveImage]

We deliberately avoid replacing ComfyUI's KSampler -- standard SamplerCustom +
UniPC works once the sigmas are right and the model accepts (B, 3, H, W) input.
"""

import os
import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

import folder_paths  # ComfyUI module; only available when running inside ComfyUI

from .adapter_loader import (
    apply_structural_surgery,
    build_comfy_patches,
)
from .scheduler import compute_sigmas
from .oklab import (
    encode_image_to_oklab_latent,
    decode_oklab_latent_to_image,
)
from .key_map import (
    translate,
    ASYMFLUX_PATCH_SIZE,
    ASYMFLUX_IN_CHANNELS,
    ASYMFLUX_OUT_CHANNELS,
)


# ---------------------------------------------------------------------------
# Adapter loader
# ---------------------------------------------------------------------------

class AsymFlux2LoadAdapter:
    """Mutates a stock FLUX.2-klein-9B model into AsymFLUX.2-klein.

    The base MODEL must already be a FLUX.2 klein checkpoint loaded via the
    standard UNETLoader / CheckpointLoaderSimple. This node performs in-place
    surgery: replaces img_in / final_layer.linear with new shapes, overwrites
    final_layer.adaLN_modulation, registers proj_buffer/scale_buffer, and
    fuses the rank-256 LoRA deltas into 58 weight tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # The adapter safetensors should sit in models/asymflux2/.
        return {
            'required': {
                'model': ('MODEL',),
                'adapter_name': (folder_paths.get_filename_list('asymflux2'),),
            },
            'optional': {
                'strict': ('BOOLEAN', {'default': True}),
            },
        }

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'AsymFlux2'

    def load(self, model, adapter_name: str, strict: bool = True):
        import time
        if load_safetensors is None:
            raise RuntimeError(
                'safetensors is not installed; run `pip install safetensors` '
                'inside the ComfyUI environment.')

        t0 = time.time()
        adapter_path = folder_paths.get_full_path('asymflux2', adapter_name)
        print(f'[AsymFlux2LoadAdapter] loading {adapter_path} ...', flush=True)
        state = load_safetensors(adapter_path)
        print(f'[AsymFlux2LoadAdapter]   safetensors loaded in {time.time()-t0:.1f}s '
              f'({len(state)} keys)', flush=True)

        plan = translate(state)
        if plan.unknown_keys:
            msg = f'unknown adapter keys ({len(plan.unknown_keys)}): {plan.unknown_keys[:5]}'
            if strict:
                raise KeyError(msg)
            print(f'[AsymFlux2LoadAdapter] WARN: {msg}', flush=True)

        # Clone the patcher so we don't clobber the user's base model.
        patched = model.clone()
        flux = patched.model.diffusion_model

        # Pick compute dtype that matches the base model.
        try:
            compute_dtype = flux.img_in.weight.dtype
        except AttributeError:
            compute_dtype = torch.bfloat16

        # 1. Structural surgery (tiny -- a few MB total, no model offload).
        t1 = time.time()
        manifest = apply_structural_surgery(flux, plan, compute_dtype)
        print(f'[AsymFlux2LoadAdapter]   surgery done in {time.time()-t1:.1f}s: '
              f'replaced={len(manifest["replaced_modules"])}, '
              f'buffers={len(manifest["registered_buffers"])}', flush=True)

        # Re-declare patch grid / channels so Flux.process_img patches correctly.
        flux.patch_size = ASYMFLUX_PATCH_SIZE
        flux.in_channels = ASYMFLUX_IN_CHANNELS * ASYMFLUX_PATCH_SIZE ** 2
        flux.out_channels = ASYMFLUX_OUT_CHANNELS * ASYMFLUX_PATCH_SIZE ** 2

        # 2. Build patches dict and register with the ModelPatcher.
        # ComfyUI applies these LAZILY during weight load, so we never have to
        # offload the 18 GB base model.
        t2 = time.time()
        patches = build_comfy_patches(plan)
        print(f'[AsymFlux2LoadAdapter]   {len(patches)} patches built in '
              f'{time.time()-t2:.1f}s (lora deltas materialized)', flush=True)

        t3 = time.time()
        patched.add_patches(patches, strength_patch=1.0, strength_model=1.0)
        print(f'[AsymFlux2LoadAdapter]   patches registered in '
              f'{time.time()-t3:.1f}s. total: {time.time()-t0:.1f}s', flush=True)

        return (patched,)


# ---------------------------------------------------------------------------
# Empty pixel-space latent
# ---------------------------------------------------------------------------

class AsymFlux2EmptyLatent:
    """3-channel pixel-resolution latent of zeros, ready to be seeded with
    noise by SamplerCustom."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'width':  ('INT', {'default': 1024, 'min': 64, 'max': 4096, 'step': 16}),
                'height': ('INT', {'default': 1024, 'min': 64, 'max': 4096, 'step': 16}),
                'batch_size': ('INT', {'default': 1, 'min': 1, 'max': 16}),
            },
        }

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'build'
    CATEGORY = 'AsymFlux2'

    def build(self, width: int, height: int, batch_size: int):
        # Width/height must be multiples of patch_size (16).
        if width % ASYMFLUX_PATCH_SIZE or height % ASYMFLUX_PATCH_SIZE:
            raise ValueError(
                f'width and height must be multiples of {ASYMFLUX_PATCH_SIZE}; '
                f'got {width}x{height}')
        samples = torch.zeros(
            (batch_size, ASYMFLUX_IN_CHANNELS, height, width), dtype=torch.float32)
        return ({'samples': samples},)


# ---------------------------------------------------------------------------
# Sigma schedule
# ---------------------------------------------------------------------------

class AsymFlux2Sigmas:
    """Produces the sqrt-shift sigmas LakonLab's FlowAdapterScheduler would
    have produced for the given image size and step count."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'steps':  ('INT', {'default': 38, 'min': 1, 'max': 500}),
                'width':  ('INT', {'default': 1024, 'min': 64, 'max': 4096}),
                'height': ('INT', {'default': 1024, 'min': 64, 'max': 4096}),
            },
            'optional': {
                'base_shift': ('FLOAT', {'default': 17.0, 'min': 0.1, 'max': 100.0}),
                'max_shift':  ('FLOAT', {'default': 34.0, 'min': 0.1, 'max': 100.0}),
            },
        }

    RETURN_TYPES = ('SIGMAS',)
    FUNCTION = 'build'
    CATEGORY = 'AsymFlux2'

    def build(self, steps: int, width: int, height: int,
              base_shift: float = 17.0, max_shift: float = 34.0):
        sigmas = compute_sigmas(
            num_inference_steps=steps,
            image_pixel_count=width * height,
            base_shift=base_shift,
            max_shift=max_shift,
        )
        return (sigmas,)


# ---------------------------------------------------------------------------
# Oklab encode / decode (replaces VAE)
# ---------------------------------------------------------------------------

class OklabEncode:
    """ComfyUI IMAGE -> 3-channel Oklab latent (after the affine norm)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',)}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'encode'
    CATEGORY = 'AsymFlux2'

    def encode(self, image: torch.Tensor):
        latent = encode_image_to_oklab_latent(image)
        return ({'samples': latent},)


class OklabDecode:
    """3-channel Oklab latent -> ComfyUI IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',)}}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'decode'
    CATEGORY = 'AsymFlux2'

    def decode(self, samples: dict):
        latent = samples['samples']
        img = decode_oklab_latent_to_image(latent)
        return (img,)


NODE_CLASS_MAPPINGS = {
    'AsymFlux2LoadAdapter': AsymFlux2LoadAdapter,
    'AsymFlux2EmptyLatent': AsymFlux2EmptyLatent,
    'AsymFlux2Sigmas':      AsymFlux2Sigmas,
    'OklabEncode':          OklabEncode,
    'OklabDecode':          OklabDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'AsymFlux2LoadAdapter': 'AsymFLUX.2 Load Adapter',
    'AsymFlux2EmptyLatent': 'AsymFLUX.2 Empty Latent (pixel)',
    'AsymFlux2Sigmas':      'AsymFLUX.2 Sigmas (sqrt-shift)',
    'OklabEncode':          'Oklab Encode (image -> latent)',
    'OklabDecode':          'Oklab Decode (latent -> image)',
}
