"""Custom latent format for AsymFLUX.2 pixel-space sampling.

Stock FLUX.2 reports `latent_channels = 128` and `spacial_downscale_ratio = 16`
because the base model operates on a 128-channel VAE latent at 1/16 image
resolution. AsymFLUX.2 operates *directly on pixels* (3 channels, full
resolution), so we need to tell ComfyUI that.

If we don't, `comfy.sample.fix_empty_latent_channels` sees a 3-channel latent
arriving for a 128-channel model and silently *tiles* it to 128 channels by
repeating, which then reaches `img_in` and blows up the shape contract.
"""

try:
    from comfy.latent_formats import LatentFormat
except ImportError:  # standalone usage (tests, type-checking)
    class LatentFormat:  # type: ignore
        latent_channels = 4
        spacial_downscale_ratio = 8
        def process_in(self, latent):
            return latent
        def process_out(self, latent):
            return latent


class AsymFlux2Pixel(LatentFormat):
    """3-channel pixel-resolution 'latent' (= image in normalized Oklab).

    Operating in pixel space means no spatial downscale and no learned VAE;
    the only transform is the deterministic affine + Oklab handled outside
    of this class by OklabEncode / OklabDecode nodes.
    """
    latent_channels = 3
    spacial_downscale_ratio = 1

    def __init__(self):
        # No latent_rgb_factors -- previews in the ComfyUI UI will look raw,
        # which is fine because Oklab Decode is the proper viewer.
        pass

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent
