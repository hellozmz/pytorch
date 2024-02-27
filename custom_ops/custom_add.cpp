#include <iostream>
#include <torch/extension.h>

torch::Tensor custom_add_forward(torch::Tensor input1, torch::Tensor input2) {
    std::cout << "call custom_add_forward." << std::endl;
    return input1 + input2;
}

std::vector<torch::Tensor> custom_add_backward(torch::Tensor grad_output) {
    std::cout << "call custom_add_backward." << std::endl;
    return {grad_output, grad_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_add_forward, "Custom Add Forward");
    m.def("backward", &custom_add_backward, "Custom Add Backward");
}
