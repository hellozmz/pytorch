from torch.autograd import Function
import torch.utils.cpp_extension

class CustomAddFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        return custom_add_cpp.forward(input1, input2)

    @staticmethod
    def backward(ctx, grad_output):
        print('CustomAddFunction run to backward.')
        input1, input2 = ctx.saved_tensors
        x1, x2 =  custom_add_cpp.backward(grad_output)
        return x1, x2

# load custom operator module
custom_add_cpp = torch.utils.cpp_extension.load(
    name='custom_add_cpp',
    sources=['custom_add.cpp'],
    extra_cflags=['-std=c++14'],
)

# get custom impl
custom_add_fwd = CustomAddFunction.apply
# custom_add_bwd = custom_add_cpp.backward

# define Python api
def custom_add(input1, input2):
    return custom_add_fwd(input1, input2)

# def custom_add_backward(input1):
#     return custom_add_bwd(input1)
