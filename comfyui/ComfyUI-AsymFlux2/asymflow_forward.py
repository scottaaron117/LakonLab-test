"""Install AsymFlow's calibration + asymmetric-velocity recovery around
ComfyUI's stock `Flux._forward`.

AsymFlow doesn't predict standard flow velocity; it predicts the *asymmetric
velocity*  u_A = P*eps - x_0  where P projects noise onto a low-rank
subspace. To use the model with a stock flow-matching sampler, every call
must:

  1. Compute   k = 1 / (s + (1 - s) * sigma)   (s = scale_buffer scalar)
     and use it to (a) scale the input  x_t -> x_t * k  and
                   (b) shift the timestep  t -> t * k.
  2. Run the transformer to produce  u_A  (asymmetric velocity).
  3. Recover the true full-rank velocity from u_A and x_t via an
     orthogonal decomposition through proj_buffer:
         u = sk * u_A_subspace + (1 - sk) / sigma * x_t_subspace
             + (x_t_complement + s * u_A_complement) / sigma
     where sk = s * k  and  *_subspace = state @ proj_buffer @ proj_buffer.T.

We monkey-patch `flux._forward` (the body executed inside ComfyUI's
WrapperExecutor) so the rest of ComfyUI's Flux pipeline (process_img,
forward_orig, unpatchify) is unchanged.
"""

import torch
from einops import rearrange


# Defaults match LakonLab's AsymFlux2Transformer2DModel.__init__:
#   sigma_min=1e-4, num_timesteps=1.
DEFAULT_SIGMA_MIN = 1e-4
DEFAULT_NUM_TIMESTEPS = 1


def install_asymflow_forward(flux,
                             patch_size: int,
                             sigma_min: float = DEFAULT_SIGMA_MIN,
                             num_timesteps: int = DEFAULT_NUM_TIMESTEPS) -> None:
    """Wrap `flux._forward` in place. Idempotent.

    Captured by closure: proj_buffer and scale_buffer (already registered on
    `flux` by adapter_loader.apply_structural_surgery), patch_size, sigma_min,
    num_timesteps.
    """
    if getattr(flux, '_asymflow_wrapped', False):
        return

    orig_forward = flux._forward

    def asymflow_forward(x, timestep, context, y=None, guidance=None,
                         ref_latents=None, control=None,
                         transformer_options={}, **kwargs):
        # --- 1. Calibration ---
        # ComfyUI passes sigma as `timestep` (flow-matching convention).
        bs = x.shape[0]
        device = x.device

        sigma = timestep.float()
        if sigma.dim() == 0:
            sigma = sigma.expand(bs)
        elif sigma.shape[0] != bs:
            sigma = sigma.expand(bs)
        sigma = sigma / num_timesteps

        s = flux.scale_buffer.float().to(device)
        k = 1.0 / (s + (1.0 - s) * sigma)             # (B,)
        cal_timestep = timestep * k                   # may differ per-batch

        # Scale x by k along the batch dim.
        k_for_x = k.to(x.dtype).view(bs, *([1] * (x.dim() - 1)))
        scaled_x = x * k_for_x

        # --- 2. Run the transformer; this returns u_A in (B, C, H, W) form ---
        u_a = orig_forward(scaled_x, cal_timestep, context,
                           y=y, guidance=guidance, ref_latents=ref_latents,
                           control=control, transformer_options=transformer_options,
                           **kwargs)

        # --- 3. Asymmetric -> true velocity, computed in fp32 on packed tokens ---
        with torch.autocast(device_type=device.type, enabled=False):
            x_t_packed = rearrange(
                x.float(), 'b c (h ph) (w pw) -> b (h w) (c ph pw)',
                ph=patch_size, pw=patch_size)
            u_a_packed = rearrange(
                u_a.float(), 'b c (h ph) (w pw) -> b (h w) (c ph pw)',
                ph=patch_size, pw=patch_size)

            proj_buffer = flux.proj_buffer.float().to(device)              # (768, 128)
            P = proj_buffer @ proj_buffer.T                                # (768, 768)

            x_t_subspace = x_t_packed @ P
            x_t_complement = x_t_packed - x_t_subspace
            u_a_subspace = u_a_packed @ P
            u_a_complement = u_a_packed - u_a_subspace

            sk = (s * k).view(bs, 1, 1)                                    # (B, 1, 1)
            sigma_clamped = sigma.clamp(min=sigma_min).view(bs, 1, 1)

            u_subspace = sk * u_a_subspace + (1.0 - sk) / sigma_clamped * x_t_subspace
            u_complement = (x_t_complement + s * u_a_complement) / sigma_clamped
            u_packed = u_subspace + u_complement

        # Unpack back to (B, C, H, W) in the same dtype the rest of the model uses.
        h_pix, w_pix = x.shape[-2:]
        h_tok, w_tok = h_pix // patch_size, w_pix // patch_size
        u = rearrange(
            u_packed.to(u_a.dtype),
            'b (h w) (c ph pw) -> b c (h ph) (w pw)',
            h=h_tok, w=w_tok, ph=patch_size, pw=patch_size)
        return u

    flux._forward = asymflow_forward
    flux._asymflow_wrapped = True
