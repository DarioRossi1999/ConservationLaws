
"""Reversible coupling ablation for HybridBackbone1d blocks.

Does not modify checkpoints, model parameters, or the original models.py code.
"""

from contextlib import contextmanager

from torch import nn


@contextmanager
def without_coupling(model: nn.Module):
    """Temporarily set c^ell to zero at the input of every mixer.

    Each mixer receives [h, g, l, c] with shape (B, 4*width, N).
    Only the last `width` channels, corresponding to coupling, are zeroed.
    """
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "mix"):
        raise TypeError("Expected a HybridDtStep1d model with backbone.mix")

    backbone = model.backbone
    width = int(backbone.width)
    mixers = backbone.mix
    if len(mixers) != int(backbone.n_layers):
        raise ValueError("The number of mixers does not match n_layers")

    def zero_c_before_mixer(module, inputs):
        if len(inputs) != 1:
            raise ValueError("Each mixer must receive exactly one input tensor")
        z = inputs[0]
        if z.ndim != 3 or z.shape[1] != 4 * width:
            raise ValueError(
                f"Invalid mixer input shape: {tuple(z.shape)}; "
                f"expected (B, {4 * width}, N)"
            )
        z_ablated = z.clone()
        z_ablated[:, 3 * width : 4 * width, :] = 0
        return (z_ablated,)

    handles = []
    try:
        for mixer in mixers:
            if not isinstance(mixer, nn.Conv1d) or mixer.in_channels != 4 * width:
                raise TypeError("Unexpected mixer: expected Conv1d with 4*width input channels")
            handles.append(mixer.register_forward_pre_hook(zero_c_before_mixer))
        yield model
    finally:
        for handle in handles:
            handle.remove()
