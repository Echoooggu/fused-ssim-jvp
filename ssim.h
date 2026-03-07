#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
);

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor, torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim_jvp(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img1_grad,
    torch::Tensor &img2,
    torch::Tensor &img2_grad,
    bool train
);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);

std::tuple<torch::Tensor, torch::Tensor>
fusedssim_backward_jvp(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img1_grad,
    torch::Tensor &img2,
    torch::Tensor &img2_grad,
    torch::Tensor &dL_dmap,
    torch::Tensor &dL_dmap_grad,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dmu1_grad,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma1_sq_grad,
    torch::Tensor &dm_dsigma12,
    torch::Tensor &dm_dsigma12_grad
);
