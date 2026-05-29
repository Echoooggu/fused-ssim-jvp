import torch
import torch.autograd.forward_ad as fwAD
from fused_ssim.utils import has_tangent, get_tangent
from fused_ssim_cuda import fusedssim, fusedssim_backward, fusedssim_jvp, fusedssim_backward_jvp

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding, train, jvp, img1_tangent, img2_tangent):

        if not jvp:
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)
        else:
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, ssim_map_tangent, dm_dmu1_tangent, dm_dsigma1_sq_tangent, dm_dsigma12_tangent = fusedssim_jvp(
                C1, C2, img1, img1_tangent, img2, img2_tangent, train
            )

            if padding == "valid":
                ssim_map_tangent = ssim_map_tangent[:, :, 5:-5, 5:-5]

            ctx.save_for_forward(ssim_map_tangent)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None, None, None, None # return nine to match the number of the inputs of forward

    @staticmethod
    def jvp(ctx, grad_C1, grad_C2, grad_img1, grad_img2, grad_padding, grad_train, grad_jvp, grad_img1_tangent, grad_img2_tangent):
        (ssim_map_tangent,) = ctx.saved_tensors
        return ssim_map_tangent # return one to match the number of the output of forward

def fused_ssim_per_pixel(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    tensor_args = (img1, img2)

    jvp = any(has_tangent(x) for x in tensor_args)

    if not jvp:
        ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, jvp, None, None)
    else:
        img1_tangent = get_tangent(img1)
        img2_tangent = get_tangent(img2)
        ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, jvp, img1_tangent, img2_tangent)

    return ssim_map

def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    tensor_args = (img1, img2)

    jvp = any(has_tangent(x) for x in tensor_args)

    if not jvp:
        ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, jvp, None, None)
    else:
        img1_tangent = get_tangent(img1)
        img2_tangent = get_tangent(img2)
        ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, jvp, img1_tangent, img2_tangent)

    return ssim_map.mean()
