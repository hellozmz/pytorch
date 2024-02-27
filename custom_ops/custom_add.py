import torch.utils.cpp_extension

# 加载自定义扩展模块
custom_add_cpp = torch.utils.cpp_extension.load(
    name='custom_add_cpp',
    sources=['custom_add.cpp'],
    extra_cflags=['-std=c++14'],
)

# 获取自定义算子函数
custom_add_fwd = custom_add_cpp.forward
custom_add_bwd = custom_add_cpp.backward

# 定义 Python 接口
def custom_add(input1, input2):
    return custom_add_fwd(input1, input2)

def custom_add_backward(input1):
    return custom_add_bwd(input1)
