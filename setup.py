from setuptools import setup, Extension
from setuptools import find_packages
import pybind11

ext_modules = [
    Extension(
        "components.tokenizer_cpp",
        sources = ["components/tokenizer_cpp/bl_bpe_tokenizer.cpp", "components/tokenizer_cpp/bindings.cpp"],
        include_dirs = [pybind11.get_include()],
        language = "c++",
        extra_compile_args=["-O3", "-std=c++20", "-Xpreprocessor", "-fopenmp"],
        extra_link_args=["-lomp"]
    ),
]

setup(
    name = "components.tokenizer_cpp",
    version = "0.1.0",
    ext_modules = ext_modules,
    packages = ["components"],
    zip_safe = False,
)