"""
This test is used to verify the correctness of the JVP implementation of the fused_ssim_per_pixel function.
It may not pass for randomly initialized pixel values of img1, img2, and tangent_img1 but should pass for low powers of 2 (current code) and all 1s.

To run the test, simply run `python tests/test_jvp.py` from the fused-ssim root directory.
"""
import sys
from pathlib import Path

import torch
import torch.autograd.forward_ad as fwAD
from torch.testing import assert_close

# Repo root (gaussian-splatting-lm) for utils.loss_utils
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fused_ssim import fused_ssim_per_pixel
from utils.loss_utils import ssim_per_pixel


def random_low_powers_of_2(shape, device):
    powers = torch.tensor([2.0, 4.0, 8.0], device=device)
    idx = torch.randint(0, len(powers), shape, device=device)
    return powers[idx]


def ones(shape, device):
    return torch.ones(shape, device=device)


def jvp_wrt_img1(fn, img1, img2, tangent_img1):
    with fwAD.dual_level():
        img1_dual = fwAD.make_dual(img1, tangent_img1)
        out = fn(img1_dual, img2)
        primal, deriv = fwAD.unpack_dual(out)
    if deriv is None:
        deriv = torch.zeros_like(primal)
    return primal, deriv


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for fused_ssim")

    device = "cuda"
    torch.manual_seed(0)

    shape = (1, 3, 32, 32)
    img1 = random_low_powers_of_2(shape, device=device)
    img2 = random_low_powers_of_2(shape, device=device)
    tangent_img1 = random_low_powers_of_2(shape, device=device)

    # Run the reference and fused SSIM implementations with JVP
    p_ref, t_ref = jvp_wrt_img1(ssim_per_pixel, img1, img2, tangent_img1)
    p_fused, t_fused = jvp_wrt_img1(fused_ssim_per_pixel, img1, img2, tangent_img1)

    print("p_ref: ",  p_ref, "\n p_fused: ", p_fused)
    print("t_ref: ",  t_ref, "\n t_fused: ", t_fused)

    # # (1) primal = primal
    assert_close(p_ref, p_fused)
    # # (2) tangent = tangent
    assert_close(t_ref, t_fused)

    print("primal:  max |ref - fused| =", (p_ref - p_fused).abs().max().item())
    print("tangent: max |ref - fused| =", (t_ref - t_fused).abs().max().item())
    print("OK: primal and tangent match.")


if __name__ == "__main__":
    main()
