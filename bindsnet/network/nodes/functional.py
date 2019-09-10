import torch

from typing import Tuple


class _SuperSpike(torch.autograd.Function):
    """
    Zenke & Ganguli 2018
    """

    scale = 10.0

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)

        out = torch.zeros_like(v)
        out[v >= 0.0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad = grad_input / (_SuperSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


super_spike = _SuperSpike.apply


def super_spike_update(
    v: torch.Tensor,
    refrac_count: torch.Tensor,
    thresh: torch.Tensor,
    reset: torch.Tensor,
    refrac: torch.Tensor,
    lbound: torch.Tensor,
    dt: torch.Tensor,
):
    """
    Spike and refractory update of a neuron
    """

    refrac_count = (refrac_count > 0).float() * (refrac_count - dt)

    # check for v >= thresh
    s = super_spike(v - thresh)
    s_ = s.byte()

    # reset voltage upon spikes
    v = torch.where(s_, reset.expand(*v.shape), v)

    # bound voltage
    v = torch.where(v < lbound, lbound.expand(*v.shape), v)

    # reset refrac on spike
    refrac_count = torch.where(s_, refrac.expand(*v.shape), refrac_count)

    # Return the states that were updated
    return s, v, refrac_count


def if_update(
    x: torch.Tensor, v: torch.Tensor, refrac_count: torch.Tensor
) -> torch.Tensor:
    """
    Update of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.

    :param x: Input voltage into the membrane
    :param v: Membrane potential
    :return: New membrane potential
    """
    dv = torch.where(
        refrac_count == 0,
        x,
        torch.zeros(1, dtype=v.dtype, device=v.device).expand(v.shape),
    )
    return v + dv
