from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    ext_modules=[
        CppExtension(
            name="njit_wrappers._bridge",
            sources=["src/njit_wrappers/csrc/_bridge.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
