# Modified for the Hexels release; upstream attribution is retained in NOTICE.
from __future__ import annotations

from importlib import import_module
import re
from typing import Any

_EXPECTED_FORWARD_TENSOR_ARGS = 15
_EXPECTED_BACKWARD_TENSOR_ARGS = 21
_EXPECTED_BACKWARD_RETURN_TENSORS = 12
_EXPECTED_FORWARD_TOTAL_ARGS = 27
_EXPECTED_BACKWARD_TOTAL_ARGS = 34


def _count_tensor_args(doc: str) -> int:
    return len(re.findall(r"arg\d+:\s*torch\.Tensor", doc))


def _count_total_args(doc: str) -> int:
    return len(re.findall(r"arg\d+:", doc))


def _count_return_tensors(doc: str) -> int | None:
    match = re.search(r"->\s*tuple\[(.*?)\]", doc, flags=re.DOTALL)
    if match is None:
        return None
    return len(re.findall(r"torch\.Tensor", match.group(1)))


def _validate_extension(ext: Any) -> None:
    fw = getattr(ext, "rasterize_hexels3d_forward", None)
    bw = getattr(ext, "rasterize_hexels3d_backward", None)
    if fw is None or bw is None:
        raise ImportError("Missing expected Hexels3D rasterizer symbols.")

    fw_doc = getattr(fw, "__doc__", "") or ""
    bw_doc = getattr(bw, "__doc__", "") or ""

    num_fw_tensor_args = _count_tensor_args(fw_doc)
    if num_fw_tensor_args != _EXPECTED_FORWARD_TENSOR_ARGS:
        raise ImportError(
            "Hexels3D CUDA extension is out of date: "
            f"expected {_EXPECTED_FORWARD_TENSOR_ARGS} tensor args for rasterize_hexels3d_forward, "
            f"got {num_fw_tensor_args}."
        )

    num_fw_total_args = _count_total_args(fw_doc)
    if num_fw_total_args != _EXPECTED_FORWARD_TOTAL_ARGS:
        raise ImportError(
            "Hexels3D CUDA extension is out of date: "
            f"expected {_EXPECTED_FORWARD_TOTAL_ARGS} total args for rasterize_hexels3d_forward, "
            f"got {num_fw_total_args}."
        )

    num_bw_tensor_args = _count_tensor_args(bw_doc)
    if num_bw_tensor_args != _EXPECTED_BACKWARD_TENSOR_ARGS:
        raise ImportError(
            "Hexels3D CUDA extension is out of date: "
            f"expected {_EXPECTED_BACKWARD_TENSOR_ARGS} tensor args for rasterize_hexels3d_backward, "
            f"got {num_bw_tensor_args}."
        )

    num_bw_total_args = _count_total_args(bw_doc)
    if num_bw_total_args != _EXPECTED_BACKWARD_TOTAL_ARGS:
        raise ImportError(
            "Hexels3D CUDA extension is out of date: "
            f"expected {_EXPECTED_BACKWARD_TOTAL_ARGS} total args for rasterize_hexels3d_backward, "
            f"got {num_bw_total_args}."
        )

    num_bw_returns = _count_return_tensors(bw_doc)
    if (
        num_bw_returns is not None
        and num_bw_returns != _EXPECTED_BACKWARD_RETURN_TENSORS
    ):
        raise ImportError(
            "Hexels3D CUDA extension is out of date: "
            f"expected {_EXPECTED_BACKWARD_RETURN_TENSORS} return tensors from rasterize_hexels3d_backward, "
            f"got {num_bw_returns}."
        )


try:
    _C = import_module(__name__ + "._C")
except ImportError as error:
    raise ImportError(
        f"Cannot import the 3D Hexel CUDA extension `{__name__}._C`; it is not built or does not match "
        "this environment. Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` "
        "from the hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
    ) from error
_validate_extension(_C)

__all__ = ["_C"]
