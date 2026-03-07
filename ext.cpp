#include <torch/extension.h>
#include "ssim.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_jvp", &fusedssim_jvp);
  m.def("fusedssim_backward", &fusedssim_backward);
  m.def("fusedssim_backward_jvp", &fusedssim_backward_jvp);
}
