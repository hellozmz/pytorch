from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='custom_add_cpp',
    ext_modules=[
        CppExtension('custom_add_cpp', ['custom_add.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
