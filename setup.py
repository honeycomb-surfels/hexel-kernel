# Modified for the Hexels release; upstream attribution is retained in NOTICE.
import glob
import os
import os.path as osp
import platform
import sys

from setuptools import find_packages, setup

__version__ = "0.1.0"

URL = "https://github.com/honeycomb-surfels/hexel-kernel"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"

_MISSING_TORCH_HELP = """\
PyTorch is required to build the hexel-kernel CUDA extensions.

Install PyTorch first. pip's build isolation hides the installed PyTorch, so
build with isolation disabled, from the hexels checkout:
  python -m pip install --no-build-isolation submodules/hexel-kernel
or from this repository:
  python -m pip install --no-build-isolation .
"""


def _raise_missing_torch(exc: ModuleNotFoundError) -> None:
    raise ModuleNotFoundError(_MISSING_TORCH_HELP) from exc


def get_ext():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ModuleNotFoundError as exc:
        _raise_missing_torch(exc)

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)


def get_extensions():
    try:
        import torch
        from torch.__config__ import parallel_info
        from torch.utils.cpp_extension import CUDAExtension
    except ModuleNotFoundError as exc:
        _raise_missing_torch(exc)

    extensions_dir = osp.join("gsplat", "cuda", "csrc")
    sources = glob.glob(osp.join(extensions_dir, "*.cu")) + glob.glob(
        osp.join(extensions_dir, "*.cpp")
    )

    # remove generated 'hip' files, in case of rebuilds
    sources = [path for path in sources if "hip" not in path]

    undef_macros = []
    define_macros = []

    if sys.platform == "win32":
        define_macros += [("gsplat_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]

    extension = CUDAExtension(
        f"gsplat.csrc",
        sources,
        include_dirs=[osp.join(extensions_dir, "third_party", "glm")],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    hexels3d_dir = osp.join("gsplat", "cuda_hexels3d", "csrc")
    hexels3d_sources = glob.glob(osp.join(hexels3d_dir, "*.cu")) + glob.glob(
        osp.join(hexels3d_dir, "*.cpp")
    )
    hexels3d_sources = [path for path in hexels3d_sources if "hip" not in path]

    hexels3d_extension = CUDAExtension(
        "gsplat.cuda_hexels3d._C",
        hexels3d_sources,
        include_dirs=[osp.join(extensions_dir, "third_party", "glm")],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension, hexels3d_extension]


setup(
    name="hexel-kernel",
    license_files=["LICENSE", "NOTICE", "licenses/*.txt", "gsplat/cuda/csrc/third_party/glm/copying.txt"],
    version=__version__,
    description="Differentiable 2D and 3D Hexel rasterizers, including Image-GS",
    keywords="gaussian, splatting, cuda",
    url=URL,
    python_requires=">=3.11",
    install_requires=[
        "jaxtyping",
        "torch",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
            "ninja",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)
