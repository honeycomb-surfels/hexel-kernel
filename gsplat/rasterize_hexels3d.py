from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch import Tensor
from torch.autograd import Function

ColorModel = Literal["flat", "linear", "quadratic", "quadratic_bounded"]
ViewModel = Literal["none", "linear", "lobe"]
ExactSortMode = Literal[
    "center_replay", "hybrid", "selected_all", "crop_all", "boundary_tail"
]

_COLOR_MODEL_TO_INT: dict[str, int] = {
    "flat": 0,
    "linear": 1,
    "quadratic": 2,
    "quadratic_bounded": 3,
}
_VIEW_MODEL_TO_INT: dict[str, int] = {"none": 0, "linear": 1, "lobe": 2}
_EXACT_SORT_MODE_TO_INT: dict[str, int] = {
    "center_replay": 0,
    "hybrid": 1,
    "selected_all": 2,
    "crop_all": 3,
    "boundary_tail": 4,
}


def _color_model_to_int(color_model: ColorModel | int) -> int:
    if isinstance(color_model, int):
        if color_model in (0, 1, 2, 3):
            return int(color_model)
        raise ValueError(
            "color_model int must be one of: 0 (flat), 1 (linear), "
            "2 (quadratic), 3 (quadratic_bounded)"
        )
    if color_model not in _COLOR_MODEL_TO_INT:
        raise ValueError(
            "color_model must be one of: flat, linear, quadratic, quadratic_bounded"
        )
    return _COLOR_MODEL_TO_INT[color_model]


def _view_model_to_int(view_model: ViewModel | int) -> int:
    if isinstance(view_model, int):
        if view_model in (0, 1, 2):
            return int(view_model)
        raise ValueError("view_model int must be one of: 0 (none), 1 (linear), 2 (lobe)")
    if view_model not in _VIEW_MODEL_TO_INT:
        raise ValueError("view_model must be one of: none, linear, lobe")
    return _VIEW_MODEL_TO_INT[view_model]


def _empty_like(x: Tensor) -> Tensor:
    return torch.empty((0,), device=x.device, dtype=x.dtype)


def _prepare_appearance_tensors(
    means3d: Tensor,
    color_grads: Tensor,
    color_quads: Tensor | None,
    view_grads: Tensor,
    *,
    color_model: int,
    use_view_grads: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    empty = _empty_like(means3d)
    color_grads_in = color_grads if color_model >= 1 else empty
    if color_model in (2, 3):
        if color_quads is None:
            raise ValueError("color_quads must be provided for a quadratic color model")
        color_quads_in = color_quads
    else:
        color_quads_in = empty
    view_grads_in = view_grads if use_view_grads else empty
    return color_grads_in, color_quads_in, view_grads_in


def rasterize_hexels3d_forward(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor]:
    from .cuda_hexels3d import _C

    if _C is None:
        raise RuntimeError(
            "3D Hexel CUDA extension is not available. "
            "Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)

    return _C.rasterize_hexels3d_forward(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(debug),
    )


def rasterize_hexels3d_owner_forward(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Render dominant compositing-owner diagnostics without autograd state."""
    from .cuda_hexels3d import _C

    if _C is None or not hasattr(_C, "rasterize_hexels3d_owner_forward"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_owner_forward`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)

    return _C.rasterize_hexels3d_owner_forward(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(debug),
    )


def rasterize_hexels3d_forward_with_contrib(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    from .cuda_hexels3d import _C

    if _C is None:
        raise RuntimeError(
            "3D Hexel CUDA extension is not available. "
            "Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )
    if not hasattr(_C, "rasterize_hexels3d_forward_with_contrib"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_forward_with_contrib`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)

    return _C.rasterize_hexels3d_forward_with_contrib(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(debug),
    )


def rasterize_hexels3d_forward_with_overdraw(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    overdraw_base_count: int = -1,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    from .cuda_hexels3d import _C

    if _C is None:
        raise RuntimeError(
            "3D Hexel CUDA extension is not available. "
            "Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )
    if not hasattr(_C, "rasterize_hexels3d_forward_with_overdraw"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_forward_with_overdraw`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)

    return _C.rasterize_hexels3d_forward_with_overdraw(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        int(overdraw_base_count),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(debug),
    )


def rasterize_hexels3d_forward_with_overdraw_hits(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    overdraw_base_count: int = -1,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    from .cuda_hexels3d import _C

    if _C is None:
        raise RuntimeError(
            "3D Hexel CUDA extension is not available. "
            "Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )
    if not hasattr(_C, "rasterize_hexels3d_forward_with_overdraw_hits"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_forward_with_overdraw_hits`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)

    return _C.rasterize_hexels3d_forward_with_overdraw_hits(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        int(overdraw_base_count),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(debug),
    )


def rasterize_hexels3d_forward_with_contrib_overdraw(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    overdraw_base_count: int = -1,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    from .cuda_hexels3d import _C

    if _C is None:
        raise RuntimeError(
            "3D Hexel CUDA extension is not available. "
            "Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )
    if not hasattr(_C, "rasterize_hexels3d_forward_with_contrib_overdraw"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_forward_with_contrib_overdraw`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)

    return _C.rasterize_hexels3d_forward_with_contrib_overdraw(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        int(overdraw_base_count),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(debug),
    )


def rasterize_hexels3d_forward_with_contrib_overdraw_hits(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    overdraw_base_count: int = -1,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    from .cuda_hexels3d import _C

    if _C is None:
        raise RuntimeError(
            "3D Hexel CUDA extension is not available. "
            "Build it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )
    if not hasattr(_C, "rasterize_hexels3d_forward_with_contrib_overdraw_hits"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_forward_with_contrib_overdraw_hits`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)

    return _C.rasterize_hexels3d_forward_with_contrib_overdraw_hits(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        int(overdraw_base_count),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(debug),
    )


def rasterize_hexels3d_exact_sort_forward(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    native_image: Tensor,
    selected_mask: Tensor,
    image_width: int,
    image_height: int,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    mode: ExactSortMode = "center_replay",
    crop_xyxy: tuple[int, int, int, int] | None = None,
    prefilter_edge_on: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Replay a crop; selected_mask must index the packed rasterizer tensors."""
    from .cuda_hexels3d import _C

    if _C is None or not hasattr(_C, "rasterize_hexels3d_exact_sort_forward"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_exact_sort_forward`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )
    if mode not in _EXACT_SORT_MODE_TO_INT:
        raise ValueError(f"Unknown exact-sort mode: {mode}")
    if prefilter_edge_on:
        raise ValueError("The exact-sort oracle does not support prefilter_edge_on=True")
    if crop_xyxy is None:
        crop_xyxy = (0, 0, int(image_width), int(image_height))
    if len(crop_xyxy) != 4:
        raise ValueError("crop_xyxy must contain (x0, y0, x1, y1)")

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)
    x0, y0, x1, y1 = crop_xyxy

    return _C.rasterize_hexels3d_exact_sort_forward(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        native_image.contiguous(),
        selected_mask.contiguous(),
        int(image_width),
        int(image_height),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        int(_EXACT_SORT_MODE_TO_INT[mode]),
        int(x0),
        int(y0),
        int(x1),
        int(y1),
        bool(debug),
    )


def rasterize_hexels3d_pixel_risk_forward(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    native_image: Tensor,
    pixel_mask: Tensor,
    image_width: int,
    image_height: int,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    crop_xyxy: tuple[int, int, int, int] | None = None,
    prefilter_edge_on: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return image, risk, order-change, hit-count, and enabled maps.

    Risk is independent of ``pixel_mask``; true pixels exact-sort their complete hit chain.
    """
    from .cuda_hexels3d import _C

    if _C is None or not hasattr(_C, "rasterize_hexels3d_pixel_risk_forward"):
        raise RuntimeError(
            "3D Hexel CUDA extension is missing `rasterize_hexels3d_pixel_risk_forward`. "
            "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
            "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
        )
    if prefilter_edge_on:
        raise ValueError("The pixel-risk oracle does not support prefilter_edge_on=True")
    if crop_xyxy is None:
        crop_xyxy = (0, 0, int(image_width), int(image_height))
    if len(crop_xyxy) != 4:
        raise ValueError("crop_xyxy must contain (x0, y0, x1, y1)")

    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)
    x0, y0, x1, y1 = crop_xyxy

    return _C.rasterize_hexels3d_pixel_risk_forward(
        means3d.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        native_image.contiguous(),
        pixel_mask.contiguous(),
        int(image_width),
        int(image_height),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
        int(x0),
        int(y0),
        int(x1),
        int(y1),
        bool(debug),
    )


def _call_backward(
    ctx: object,
    grad_out_color: Tensor,
    grad_out_overdraw: Tensor | None,
    grad_out_surf_depth: Tensor | None = None,
    grad_out_surf_alpha: Tensor | None = None,
    grad_out_surf_normals: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        means3d,
        means2d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
    ) = ctx.saved_tensors[:16]
    from .cuda_hexels3d import _C

    if grad_out_overdraw is None:
        grad_out_overdraw = _empty_like(grad_out_color)
    if grad_out_surf_depth is None:
        grad_out_surf_depth = _empty_like(grad_out_color)
    if grad_out_surf_alpha is None:
        grad_out_surf_alpha = _empty_like(grad_out_color)
    if grad_out_surf_normals is None:
        grad_out_surf_normals = _empty_like(grad_out_color)

    overdraw_base_count = int(getattr(ctx, "overdraw_base_count", -1))
    prefilter_edge_on = bool(getattr(ctx, "prefilter_edge_on", False))
    conservative_bounds = ctx.conservative_bounds
    saved = ctx.saved_tensors[16:]
    backward = _C.rasterize_hexels3d_backward_cached if saved else _C.rasterize_hexels3d_backward
    return backward(
        means3d,
        means2d,
        scales,
        rotations,
        colors,
        color_grads,
        color_quads,
        view_grads,
        specular_ks,
        specular_log_s,
        opacities,
        softnesses,
        viewmatrix,
        projmatrix,
        campos,
        background,
        grad_out_color.contiguous(),
        grad_out_overdraw.contiguous(),
        grad_out_surf_depth.contiguous(),
        grad_out_surf_alpha.contiguous(),
        grad_out_surf_normals.contiguous(),
        int(ctx.image_width),
        int(ctx.image_height),
        float(ctx.tan_fovx),
        float(ctx.tan_fovy),
        float(ctx.softness),
        float(ctx.scale_modifier),
        int(ctx.color_model),
        int(getattr(ctx, "view_model", 1)),
        bool(ctx.use_view_grads),
        int(overdraw_base_count),
        bool(prefilter_edge_on),
        bool(conservative_bounds),
        bool(ctx.debug),
        *([list(saved)] if saved else []),
    )


def rasterize_hexels3d(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    means2d: Tensor | None = None,
    color_quads: Tensor | None = None,
    specular_ks: Tensor | None = None,
    specular_log_s: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    view_model: ViewModel | int = "linear",
    use_view_grads: bool = True,
    return_contrib: bool = False,
    return_overdraw: bool = False,
    return_overdraw_hits: bool = False,
    return_surface: bool = False,
    return_distortion: bool = False,
    return_median_depth: bool = False,
    overdraw_base_count: int = -1,
    prefilter_edge_on: bool = False,
    conservative_bounds: bool = False,
    debug: bool = False,
) -> Tuple[Tensor, ...]:
    """Rasterize with an optional screen-gradient proxy, not a position input.

    ``means2d`` accepts (P,3), or (P,4) to additionally accumulate the sum of
    per-pixel XY gradient norms in ``means2d.grad[:,3]``. The extra statistic
    is not a derivative of the forward image. Columns 0:3 keep the existing
    signed XY proxy and zero Z; rendered images and parameter VJPs are unchanged.

    ``conservative_bounds=True`` restores near-plane support omitted by legacy
    tile bounds, which can change existing checkpoint renders. The default
    preserves historical coverage while removing tiles that cannot contribute.
    RGB autograd calls cache forward state; inference does not retain it.
    """
    color_model_i = _color_model_to_int(color_model)
    view_model_i = _view_model_to_int(view_model)
    use_view_grads_eff = bool(use_view_grads) and view_model_i == 1
    color_grads_in, color_quads_in, view_grads_in = _prepare_appearance_tensors(
        means3d,
        color_grads,
        color_quads,
        view_grads,
        color_model=color_model_i,
        use_view_grads=use_view_grads_eff,
    )
    specular_ks_in = _empty_like(means3d)
    specular_log_s_in = _empty_like(means3d)
    if view_model_i == 2:
        if specular_ks is None or specular_log_s is None:
            raise ValueError("specular_ks and specular_log_s must be provided when view_model='lobe'")
        specular_ks_in = specular_ks
        specular_log_s_in = specular_log_s
    softnesses_in = softnesses.contiguous() if softnesses is not None else _empty_like(means3d)
    means2d_in = means2d.contiguous() if means2d is not None else _empty_like(means3d)

    args_common = (
        means3d.contiguous(),
        means2d_in,
        scales.contiguous(),
        rotations.contiguous(),
        colors.contiguous(),
        color_grads_in.contiguous(),
        color_quads_in.contiguous(),
        view_grads_in.contiguous(),
        specular_ks_in.contiguous(),
        specular_log_s_in.contiguous(),
        opacities.contiguous(),
        softnesses_in.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        background.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(softness),
        float(scale_modifier),
        int(color_model_i),
        int(view_model_i),
        bool(use_view_grads_eff),
    )
    if return_overdraw_hits and not return_overdraw:
        raise ValueError("return_overdraw_hits requires return_overdraw=True")
    if return_distortion and not return_surface:
        raise ValueError("return_distortion requires return_surface=True")
    if return_median_depth and not return_surface:
        raise ValueError("Median depth requires surface outputs")
    if return_surface:
        args = args_common + (
            bool(return_contrib),
            bool(return_overdraw),
            bool(return_overdraw_hits),
            True,
            int(overdraw_base_count),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
            bool(return_median_depth),
        )
        surface_outputs = _RasterizeHexels3DAll.apply(*args)
        (
            out_color,
            radii,
            contrib,
            overdraw,
            overdraw_hits,
            surf_depth,
            surf_alpha,
            surf_maps,
        ) = surface_outputs

        outputs: list[Tensor] = [out_color, radii]
        if return_contrib:
            outputs.append(contrib)
        if return_overdraw:
            outputs.append(overdraw)
        if return_overdraw_hits:
            outputs.append(overdraw_hits)
        outputs.extend([surf_depth, surf_alpha, surf_maps[:3]])
        if return_distortion:
            outputs.append(surf_maps[3])
        if return_median_depth:
            outputs.append(surf_maps[4])
        return tuple(outputs)
    if return_overdraw:
        args = args_common + (int(overdraw_base_count), bool(prefilter_edge_on), bool(conservative_bounds), bool(debug))
    else:
        args = args_common + (bool(prefilter_edge_on), bool(conservative_bounds), bool(debug))

    if return_contrib and return_overdraw_hits:
        return _RasterizeHexels3DWithContribOverdrawHits.apply(*args)
    if return_overdraw_hits:
        return _RasterizeHexels3DWithOverdrawHits.apply(*args)
    if return_contrib and return_overdraw:
        return _RasterizeHexels3DWithContribOverdraw.apply(*args)
    if return_contrib:
        return _RasterizeHexels3DWithContrib.apply(*args)
    if return_overdraw:
        return _RasterizeHexels3DWithOverdraw.apply(*args)
    save_state = torch.is_grad_enabled() and any(t.requires_grad for t in args_common[:16])
    return _RasterizeHexels3D.apply(*args, save_state)


def rasterize_hexels3d_with_contrib(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    use_view_grads: bool = True,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    out = rasterize_hexels3d(
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        colors=colors,
        color_grads=color_grads,
        view_grads=view_grads,
        opacities=opacities,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        campos=campos,
        background=background,
        image_width=image_width,
        image_height=image_height,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        softness=softness,
        softnesses=softnesses,
        color_quads=color_quads,
        scale_modifier=scale_modifier,
        color_model=color_model,
        use_view_grads=use_view_grads,
        return_contrib=True,
        return_overdraw=False,
        debug=debug,
    )
    out_color, radii, contrib = out
    return out_color, radii, contrib


def rasterize_hexels3d_with_overdraw(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    use_view_grads: bool = True,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    out = rasterize_hexels3d(
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        colors=colors,
        color_grads=color_grads,
        view_grads=view_grads,
        opacities=opacities,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        campos=campos,
        background=background,
        image_width=image_width,
        image_height=image_height,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        softness=softness,
        softnesses=softnesses,
        color_quads=color_quads,
        scale_modifier=scale_modifier,
        color_model=color_model,
        use_view_grads=use_view_grads,
        return_contrib=False,
        return_overdraw=True,
        debug=debug,
    )
    out_color, radii, overdraw = out
    return out_color, radii, overdraw


def rasterize_hexels3d_with_contrib_overdraw(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    view_grads: Tensor,
    opacities: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    softnesses: Tensor | None = None,
    *,
    color_quads: Tensor | None = None,
    scale_modifier: float = 1.0,
    color_model: ColorModel | int = "linear",
    use_view_grads: bool = True,
    debug: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    out = rasterize_hexels3d(
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        colors=colors,
        color_grads=color_grads,
        view_grads=view_grads,
        opacities=opacities,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        campos=campos,
        background=background,
        image_width=image_width,
        image_height=image_height,
        tan_fovx=tan_fovx,
        tan_fovy=tan_fovy,
        softness=softness,
        softnesses=softnesses,
        color_quads=color_quads,
        scale_modifier=scale_modifier,
        color_model=color_model,
        use_view_grads=use_view_grads,
        return_contrib=True,
        return_overdraw=True,
        debug=debug,
    )
    out_color, radii, contrib, overdraw = out
    return out_color, radii, contrib, overdraw


class _RasterizeHexels3D(Function):
    @staticmethod
    def forward(
        ctx: object,
        means3d: Tensor,
        means2d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_quads: Tensor,
        view_grads: Tensor,
        specular_ks: Tensor,
        specular_log_s: Tensor,
        opacities: Tensor,
        softnesses: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float,
        color_model: int,
        view_model: int,
        use_view_grads: bool,
        prefilter_edge_on: bool,
        conservative_bounds: bool,
        debug: bool,
        save_state: bool,
    ) -> Tuple[Tensor, Tensor]:
        from .cuda_hexels3d import _C

        forward = _C.rasterize_hexels3d_forward_cached if save_state else _C.rasterize_hexels3d_forward
        result = forward(
            means3d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            int(image_width),
            int(image_height),
            float(tan_fovx),
            float(tan_fovy),
            float(softness),
            float(scale_modifier),
            int(color_model),
            int(view_model),
            bool(use_view_grads),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
        )

        out_color, radii = result[:2]
        saved = result[2] if save_state else []
        ctx.save_for_backward(
            means3d,
            means2d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            *saved,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tan_fovx = float(tan_fovx)
        ctx.tan_fovy = float(tan_fovy)
        ctx.softness = float(softness)
        ctx.scale_modifier = float(scale_modifier)
        ctx.color_model = int(color_model)
        ctx.view_model = int(view_model)
        ctx.use_view_grads = bool(use_view_grads)
        ctx.prefilter_edge_on = bool(prefilter_edge_on)
        ctx.conservative_bounds = bool(conservative_bounds)
        ctx.debug = bool(debug)
        return out_color, radii

    @staticmethod
    def backward(
        ctx: object,
        grad_out_color: Tensor,
        grad_out_radii: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
        ) = _call_backward(ctx, grad_out_color, None)

        return (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # background
            None,  # image_width
            None,  # image_height
            None,  # tan_fovx
            None,  # tan_fovy
            None,  # softness
            None,  # scale_modifier
            None,  # color_model
            None,  # view_model
            None,  # use_view_grads
            None,  # prefilter_edge_on
            None,  # conservative_bounds
            None,  # debug
            None,  # save_state
        )


class _RasterizeHexels3DWithContrib(Function):
    @staticmethod
    def forward(
        ctx: object,
        means3d: Tensor,
        means2d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_quads: Tensor,
        view_grads: Tensor,
        specular_ks: Tensor,
        specular_log_s: Tensor,
        opacities: Tensor,
        softnesses: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float,
        color_model: int,
        view_model: int,
        use_view_grads: bool,
        prefilter_edge_on: bool,
        conservative_bounds: bool,
        debug: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        from .cuda_hexels3d import _C

        out_color, radii, contrib = _C.rasterize_hexels3d_forward_with_contrib(
            means3d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            int(image_width),
            int(image_height),
            float(tan_fovx),
            float(tan_fovy),
            float(softness),
            float(scale_modifier),
            int(color_model),
            int(view_model),
            bool(use_view_grads),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
        )

        ctx.save_for_backward(
            means3d,
            means2d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tan_fovx = float(tan_fovx)
        ctx.tan_fovy = float(tan_fovy)
        ctx.softness = float(softness)
        ctx.scale_modifier = float(scale_modifier)
        ctx.color_model = int(color_model)
        ctx.view_model = int(view_model)
        ctx.use_view_grads = bool(use_view_grads)
        ctx.prefilter_edge_on = bool(prefilter_edge_on)
        ctx.conservative_bounds = bool(conservative_bounds)
        ctx.debug = bool(debug)
        return out_color, radii, contrib

    @staticmethod
    def backward(
        ctx: object,
        grad_out_color: Tensor,
        grad_out_radii: Tensor | None = None,
        grad_out_contrib: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
        ) = _call_backward(ctx, grad_out_color, None)

        return (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # background
            None,  # image_width
            None,  # image_height
            None,  # tan_fovx
            None,  # tan_fovy
            None,  # softness
            None,  # scale_modifier
            None,  # color_model
            None,  # view_model
            None,  # use_view_grads
            None,  # prefilter_edge_on
            None,  # conservative_bounds
            None,  # debug
        )


class _RasterizeHexels3DWithOverdraw(Function):
    @staticmethod
    def forward(
        ctx: object,
        means3d: Tensor,
        means2d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_quads: Tensor,
        view_grads: Tensor,
        specular_ks: Tensor,
        specular_log_s: Tensor,
        opacities: Tensor,
        softnesses: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float,
        color_model: int,
        view_model: int,
        use_view_grads: bool,
        overdraw_base_count: int,
        prefilter_edge_on: bool,
        conservative_bounds: bool,
        debug: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        from .cuda_hexels3d import _C

        out_color, radii, overdraw = _C.rasterize_hexels3d_forward_with_overdraw(
            means3d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            int(image_width),
            int(image_height),
            float(tan_fovx),
            float(tan_fovy),
            float(softness),
            float(scale_modifier),
            int(color_model),
            int(view_model),
            bool(use_view_grads),
            int(overdraw_base_count),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
        )

        ctx.save_for_backward(
            means3d,
            means2d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tan_fovx = float(tan_fovx)
        ctx.tan_fovy = float(tan_fovy)
        ctx.softness = float(softness)
        ctx.scale_modifier = float(scale_modifier)
        ctx.color_model = int(color_model)
        ctx.view_model = int(view_model)
        ctx.use_view_grads = bool(use_view_grads)
        ctx.overdraw_base_count = int(overdraw_base_count)
        ctx.prefilter_edge_on = bool(prefilter_edge_on)
        ctx.conservative_bounds = bool(conservative_bounds)
        ctx.debug = bool(debug)
        return out_color, radii, overdraw

    @staticmethod
    def backward(
        ctx: object,
        grad_out_color: Tensor,
        grad_out_radii: Tensor | None = None,
        grad_out_overdraw: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
        ) = _call_backward(ctx, grad_out_color, grad_out_overdraw)

        return (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # background
            None,  # image_width
            None,  # image_height
            None,  # tan_fovx
            None,  # tan_fovy
            None,  # softness
            None,  # scale_modifier
            None,  # color_model
            None,  # view_model
            None,  # use_view_grads
            None,  # overdraw_base_count
            None,  # prefilter_edge_on
            None,  # conservative_bounds
            None,  # debug
        )


class _RasterizeHexels3DWithContribOverdraw(Function):
    @staticmethod
    def forward(
        ctx: object,
        means3d: Tensor,
        means2d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_quads: Tensor,
        view_grads: Tensor,
        specular_ks: Tensor,
        specular_log_s: Tensor,
        opacities: Tensor,
        softnesses: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float,
        color_model: int,
        view_model: int,
        use_view_grads: bool,
        overdraw_base_count: int,
        prefilter_edge_on: bool,
        conservative_bounds: bool,
        debug: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        from .cuda_hexels3d import _C

        out_color, radii, contrib, overdraw = _C.rasterize_hexels3d_forward_with_contrib_overdraw(
            means3d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            int(image_width),
            int(image_height),
            float(tan_fovx),
            float(tan_fovy),
            float(softness),
            float(scale_modifier),
            int(color_model),
            int(view_model),
            bool(use_view_grads),
            int(overdraw_base_count),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
        )

        ctx.save_for_backward(
            means3d,
            means2d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tan_fovx = float(tan_fovx)
        ctx.tan_fovy = float(tan_fovy)
        ctx.softness = float(softness)
        ctx.scale_modifier = float(scale_modifier)
        ctx.color_model = int(color_model)
        ctx.view_model = int(view_model)
        ctx.use_view_grads = bool(use_view_grads)
        ctx.overdraw_base_count = int(overdraw_base_count)
        ctx.prefilter_edge_on = bool(prefilter_edge_on)
        ctx.conservative_bounds = bool(conservative_bounds)
        ctx.debug = bool(debug)
        return out_color, radii, contrib, overdraw

    @staticmethod
    def backward(
        ctx: object,
        grad_out_color: Tensor,
        grad_out_radii: Tensor | None = None,
        grad_out_contrib: Tensor | None = None,
        grad_out_overdraw: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
        ) = _call_backward(ctx, grad_out_color, grad_out_overdraw)

        return (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # background
            None,  # image_width
            None,  # image_height
            None,  # tan_fovx
            None,  # tan_fovy
            None,  # softness
            None,  # scale_modifier
            None,  # color_model
            None,  # view_model
            None,  # use_view_grads
            None,  # overdraw_base_count
            None,  # prefilter_edge_on
            None,  # conservative_bounds
            None,  # debug
        )


class _RasterizeHexels3DWithOverdrawHits(Function):
    @staticmethod
    def forward(
        ctx: object,
        means3d: Tensor,
        means2d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_quads: Tensor,
        view_grads: Tensor,
        specular_ks: Tensor,
        specular_log_s: Tensor,
        opacities: Tensor,
        softnesses: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float,
        color_model: int,
        view_model: int,
        use_view_grads: bool,
        overdraw_base_count: int,
        prefilter_edge_on: bool,
        conservative_bounds: bool,
        debug: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        from .cuda_hexels3d import _C

        out_color, radii, overdraw, overdraw_hits = _C.rasterize_hexels3d_forward_with_overdraw_hits(
            means3d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            int(image_width),
            int(image_height),
            float(tan_fovx),
            float(tan_fovy),
            float(softness),
            float(scale_modifier),
            int(color_model),
            int(view_model),
            bool(use_view_grads),
            int(overdraw_base_count),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
        )

        ctx.save_for_backward(
            means3d,
            means2d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tan_fovx = float(tan_fovx)
        ctx.tan_fovy = float(tan_fovy)
        ctx.softness = float(softness)
        ctx.scale_modifier = float(scale_modifier)
        ctx.color_model = int(color_model)
        ctx.view_model = int(view_model)
        ctx.use_view_grads = bool(use_view_grads)
        ctx.overdraw_base_count = int(overdraw_base_count)
        ctx.prefilter_edge_on = bool(prefilter_edge_on)
        ctx.conservative_bounds = bool(conservative_bounds)
        ctx.debug = bool(debug)
        return out_color, radii, overdraw, overdraw_hits

    @staticmethod
    def backward(
        ctx: object,
        grad_out_color: Tensor,
        grad_out_radii: Tensor | None = None,
        grad_out_overdraw: Tensor | None = None,
        grad_out_overdraw_hits: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
        ) = _call_backward(ctx, grad_out_color, grad_out_overdraw)

        return (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # background
            None,  # image_width
            None,  # image_height
            None,  # tan_fovx
            None,  # tan_fovy
            None,  # softness
            None,  # scale_modifier
            None,  # color_model
            None,  # view_model
            None,  # use_view_grads
            None,  # overdraw_base_count
            None,  # prefilter_edge_on
            None,  # conservative_bounds
            None,  # debug
        )


class _RasterizeHexels3DWithContribOverdrawHits(Function):
    @staticmethod
    def forward(
        ctx: object,
        means3d: Tensor,
        means2d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_quads: Tensor,
        view_grads: Tensor,
        specular_ks: Tensor,
        specular_log_s: Tensor,
        opacities: Tensor,
        softnesses: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float,
        color_model: int,
        view_model: int,
        use_view_grads: bool,
        overdraw_base_count: int,
        prefilter_edge_on: bool,
        conservative_bounds: bool,
        debug: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        from .cuda_hexels3d import _C

        out_color, radii, contrib, overdraw, overdraw_hits = _C.rasterize_hexels3d_forward_with_contrib_overdraw_hits(
            means3d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            int(image_width),
            int(image_height),
            float(tan_fovx),
            float(tan_fovy),
            float(softness),
            float(scale_modifier),
            int(color_model),
            int(view_model),
            bool(use_view_grads),
            int(overdraw_base_count),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
        )

        ctx.save_for_backward(
            means3d,
            means2d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tan_fovx = float(tan_fovx)
        ctx.tan_fovy = float(tan_fovy)
        ctx.softness = float(softness)
        ctx.scale_modifier = float(scale_modifier)
        ctx.color_model = int(color_model)
        ctx.view_model = int(view_model)
        ctx.use_view_grads = bool(use_view_grads)
        ctx.overdraw_base_count = int(overdraw_base_count)
        ctx.prefilter_edge_on = bool(prefilter_edge_on)
        ctx.conservative_bounds = bool(conservative_bounds)
        ctx.debug = bool(debug)
        return out_color, radii, contrib, overdraw, overdraw_hits

    @staticmethod
    def backward(
        ctx: object,
        grad_out_color: Tensor,
        grad_out_radii: Tensor | None = None,
        grad_out_contrib: Tensor | None = None,
        grad_out_overdraw: Tensor | None = None,
        grad_out_overdraw_hits: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
        ) = _call_backward(ctx, grad_out_color, grad_out_overdraw)

        return (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # background
            None,  # image_width
            None,  # image_height
            None,  # tan_fovx
            None,  # tan_fovy
            None,  # softness
            None,  # scale_modifier
            None,  # color_model
            None,  # view_model
            None,  # use_view_grads
            None,  # overdraw_base_count
            None,  # prefilter_edge_on
            None,  # conservative_bounds
            None,  # debug
        )


class _RasterizeHexels3DAll(Function):
    @staticmethod
    def forward(
        ctx: object,
        means3d: Tensor,
        means2d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_quads: Tensor,
        view_grads: Tensor,
        specular_ks: Tensor,
        specular_log_s: Tensor,
        opacities: Tensor,
        softnesses: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float,
        color_model: int,
        view_model: int,
        use_view_grads: bool,
        return_contrib: bool,
        return_overdraw: bool,
        return_overdraw_hits: bool,
        return_surface: bool,
        overdraw_base_count: int,
        prefilter_edge_on: bool,
        conservative_bounds: bool,
        debug: bool,
        return_median_depth: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        from .cuda_hexels3d import _C

        if _C is None or not hasattr(_C, "rasterize_hexels3d_forward_all"):
            raise RuntimeError(
                "3D Hexel CUDA extension is missing `rasterize_hexels3d_forward_all`. "
                "Rebuild it with `python -m pip install --no-build-isolation submodules/hexel-kernel` from the "
                "hexels checkout, or `python -m pip install --no-build-isolation .` in the hexel-kernel repository."
            )

        forward = (_C.rasterize_hexels3d_forward_median if return_median_depth
                   else _C.rasterize_hexels3d_forward_all)
        out = forward(
            means3d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
            int(image_width),
            int(image_height),
            float(tan_fovx),
            float(tan_fovy),
            float(softness),
            float(scale_modifier),
            int(color_model),
            int(view_model),
            bool(use_view_grads),
            bool(return_contrib),
            bool(return_overdraw),
            bool(return_overdraw_hits),
            bool(return_surface),
            int(overdraw_base_count),
            bool(prefilter_edge_on),
            bool(conservative_bounds),
            bool(debug),
        )
        (
            out_color,
            radii,
            contrib,
            overdraw,
            overdraw_hits,
            surf_depth,
            surf_alpha,
            surf_normals,
        ) = out

        ctx.save_for_backward(
            means3d,
            means2d,
            scales,
            rotations,
            colors,
            color_grads,
            color_quads,
            view_grads,
            specular_ks,
            specular_log_s,
            opacities,
            softnesses,
            viewmatrix,
            projmatrix,
            campos,
            background,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tan_fovx = float(tan_fovx)
        ctx.tan_fovy = float(tan_fovy)
        ctx.softness = float(softness)
        ctx.scale_modifier = float(scale_modifier)
        ctx.color_model = int(color_model)
        ctx.view_model = int(view_model)
        ctx.use_view_grads = bool(use_view_grads)
        ctx.overdraw_base_count = int(overdraw_base_count)
        ctx.prefilter_edge_on = bool(prefilter_edge_on)
        ctx.conservative_bounds = bool(conservative_bounds)
        ctx.debug = bool(debug)
        return out_color, radii, contrib, overdraw, overdraw_hits, surf_depth, surf_alpha, surf_normals

    @staticmethod
    def backward(
        ctx: object,
        grad_out_color: Tensor,
        grad_out_radii: Tensor | None = None,
        grad_out_contrib: Tensor | None = None,
        grad_out_overdraw: Tensor | None = None,
        grad_out_overdraw_hits: Tensor | None = None,
        grad_out_surf_depth: Tensor | None = None,
        grad_out_surf_alpha: Tensor | None = None,
        grad_out_surf_normals: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
        ) = _call_backward(
            ctx,
            grad_out_color,
            grad_out_overdraw,
            grad_out_surf_depth,
            grad_out_surf_alpha,
            grad_out_surf_normals,
        )

        return (
            grad_means3d,
            grad_means2d,
            grad_scales,
            grad_rotations,
            grad_colors,
            grad_color_grads,
            grad_color_quads,
            grad_view_grads,
            grad_specular_ks,
            grad_specular_log_s,
            grad_opacities,
            grad_softnesses,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # background
            None,  # image_width
            None,  # image_height
            None,  # tan_fovx
            None,  # tan_fovy
            None,  # softness
            None,  # scale_modifier
            None,  # color_model
            None,  # view_model
            None,  # use_view_grads
            None,  # return_contrib
            None,  # return_overdraw
            None,  # return_overdraw_hits
            None,  # return_surface
            None,  # overdraw_base_count
            None,  # prefilter_edge_on
            None,  # conservative_bounds
            None,  # debug
            None,
        )


class Hexels3DRasterizer(torch.nn.Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        tan_fovx: float,
        tan_fovy: float,
        softness: float,
        scale_modifier: float = 1.0,
        *,
        color_model: ColorModel = "linear",
        view_model: ViewModel = "linear",
        use_view_grads: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.tan_fovx = float(tan_fovx)
        self.tan_fovy = float(tan_fovy)
        self.softness = float(softness)
        self.scale_modifier = float(scale_modifier)
        self.color_model = str(color_model)
        self.view_model = str(view_model)
        self.use_view_grads = bool(use_view_grads)
        self.debug = bool(debug)

    def render_owner(
        self,
        means3d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        view_grads: Tensor,
        opacities: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        *,
        color_quads: Tensor | None = None,
        specular_ks: Tensor | None = None,
        specular_log_s: Tensor | None = None,
        softnesses: Tensor | None = None,
        prefilter_edge_on: bool = False,
        conservative_bounds: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return rasterize_hexels3d_owner_forward(
            means3d=means3d,
            scales=scales,
            rotations=rotations,
            colors=colors,
            color_grads=color_grads,
            color_quads=color_quads,
            view_grads=view_grads,
            specular_ks=specular_ks,
            specular_log_s=specular_log_s,
            opacities=opacities,
            softnesses=softnesses,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            campos=campos,
            background=background,
            image_width=self.image_width,
            image_height=self.image_height,
            tan_fovx=self.tan_fovx,
            tan_fovy=self.tan_fovy,
            softness=self.softness,
            scale_modifier=self.scale_modifier,
            color_model=self.color_model,
            view_model=self.view_model,
            use_view_grads=self.use_view_grads,
            prefilter_edge_on=prefilter_edge_on,
            conservative_bounds=conservative_bounds,
            debug=self.debug,
        )

    def forward(
        self,
        means3d: Tensor,
        scales: Tensor,
        rotations: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        view_grads: Tensor,
        opacities: Tensor,
        viewmatrix: Tensor,
        projmatrix: Tensor,
        campos: Tensor,
        background: Tensor,
        *,
        means2d: Tensor | None = None,
        color_quads: Tensor | None = None,
        specular_ks: Tensor | None = None,
        specular_log_s: Tensor | None = None,
        softnesses: Tensor | None = None,
        return_contrib: bool = False,
        return_overdraw: bool = False,
        return_overdraw_hits: bool = False,
        return_surface: bool = False,
        return_distortion: bool = False,
        return_median_depth: bool = False,
        overdraw_base_count: int = -1,
        prefilter_edge_on: bool = False,
        conservative_bounds: bool = False,
    ) -> Tuple[Tensor, ...]:
        return rasterize_hexels3d(
            means3d=means3d,
            means2d=means2d,
            scales=scales,
            rotations=rotations,
            colors=colors,
            color_grads=color_grads,
            color_quads=color_quads,
            view_grads=view_grads,
            specular_ks=specular_ks,
            specular_log_s=specular_log_s,
            opacities=opacities,
            softnesses=softnesses,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            campos=campos,
            background=background,
            image_width=self.image_width,
            image_height=self.image_height,
            tan_fovx=self.tan_fovx,
            tan_fovy=self.tan_fovy,
            softness=self.softness,
            scale_modifier=self.scale_modifier,
            color_model=self.color_model,
            view_model=self.view_model,
            use_view_grads=self.use_view_grads,
            return_contrib=return_contrib,
            return_overdraw=return_overdraw,
            return_overdraw_hits=return_overdraw_hits,
            return_surface=return_surface,
            return_distortion=return_distortion,
            return_median_depth=return_median_depth,
            overdraw_base_count=overdraw_base_count,
            prefilter_edge_on=prefilter_edge_on,
            conservative_bounds=conservative_bounds,
            debug=self.debug,
        )


__all__ = [
    "Hexels3DRasterizer",
    "rasterize_hexels3d",
    "rasterize_hexels3d_owner_forward",
    "rasterize_hexels3d_forward",
    "rasterize_hexels3d_forward_with_contrib",
    "rasterize_hexels3d_forward_with_overdraw",
    "rasterize_hexels3d_forward_with_overdraw_hits",
    "rasterize_hexels3d_forward_with_contrib_overdraw",
    "rasterize_hexels3d_forward_with_contrib_overdraw_hits",
    "rasterize_hexels3d_exact_sort_forward",
    "rasterize_hexels3d_pixel_risk_forward",
    "rasterize_hexels3d_with_contrib",
    "rasterize_hexels3d_with_overdraw",
    "rasterize_hexels3d_with_contrib_overdraw",
]
