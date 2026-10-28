# Modified for the Hexels release; upstream attribution is retained in NOTICE.
"""Load the extension built from this release; never reuse a JIT cache."""

try:
    from gsplat import csrc as _C
except ImportError as error:
    raise ImportError(
        "Cannot import the 2D Hexel and Image-GS CUDA extension `gsplat.csrc`; it is not built or does not match "
        "this environment. Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` "
        "from the hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
    ) from error

__all__ = ["_C"]
