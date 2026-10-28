# Hexel kernel

CUDA rasterizers for 2D image Hexels, 3D Hexel surfels, and native Image-GS.
The package installs as `hexel-kernel` and exposes the historical `gsplat`
Python import. The [main repository](https://github.com/honeycomb-surfels/hexels)
provides trainers, evaluation, surface extraction and experiment scripts.

## Build

Use Linux, Python 3.11, PyTorch 2.5.1 with CUDA 12.1, an NVIDIA GPU, and a
CUDA toolkit with `nvcc`. The tested build uses toolkit 12.3 and RTX A5000
(sm_86). PyTorch's runtime packages alone do not include the compiler.
A host C++ compiler and driver compatible with the toolkit are required.
Other GPU architectures need their own `TORCH_CUDA_ARCH_LIST` and rebuild.
No smaller GPU-memory minimum has been validated; see the main sample report.

```bash
export CUDA_HOME=/path/to/cuda-12.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
python -m pip install --no-build-isolation .
```

Run builds and CUDA tests on a GPU machine or Slurm GPU allocation.
`pip install --no-build-isolation -e .` is also available for development.
Do not install an unrelated `gsplat` distribution into the same environment.

## Scope and compatibility

Image-GS and 3D Hexels compile into separate extensions. Including Image-GS
adds build time and installed bytes; it does not add rendering passes to 3D
training or inference. The 2D sources match the audited image experiment
runtime; 3D CUDA arithmetic is preserved from the research checkout. That
checkout includes a correction to the optional edge-prefilter projection
(`prefilter_edge_on`) made after the paper binary was built; all 198 inspected
paper training configs disable edge prefiltering. Bit-level parity with the
paper binary was checked only with CUDA 12.3 on sm_86.
Quaternions use WXYZ and geometric tensors use contiguous CUDA float32.

The low-level `prefilter_edge_on` argument remains for historical API and viewer
regression coverage; it is not enabled by a release training recipe. Diagnostic/oracle
entry points remain opt-in and are not executed by the standard render path.
Implicit JIT fallbacks were removed: importing a different cached binary can
no longer conceal a missing or incompatible release build.

## Attribution

This package adapts [NYU-ICL/Image-GS](https://github.com/NYU-ICL/image-gs),
revision `03088368d42684fb54225c981cfd94b58cc0393a`, whose gsplat code derives
from [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat).
The original [Image-GS MIT notice](licenses/Image-GS.txt) and inherited
[gsplat Apache-2.0 license](licenses/gsplat-Apache-2.0.txt) are retained.
See [NOTICE](NOTICE) for the source and modification attribution.
The bundled GLM headers retain their [upstream license](gsplat/cuda/csrc/third_party/glm/copying.txt).
Original Hexel additions use the root [MIT license](LICENSE); these notices
remain applicable to the corresponding third-party files.
