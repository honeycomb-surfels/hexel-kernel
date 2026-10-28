#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    m.def("nd_rasterize_forward", &nd_rasterize_forward_tensor);
    m.def("nd_rasterize_backward", &nd_rasterize_backward_tensor);

    m.def("nd_rasterize_forward_topk_norm",  &nd_rasterize_forward_topk_norm_tensor);
    m.def("nd_rasterize_backward_topk_norm", &nd_rasterize_backward_topk_norm_tensor);
    m.def("nd_rasterize_forward_topk_norm_stats", &nd_rasterize_forward_topk_norm_stats_tensor);
    m.def("nd_rasterize_backward_topk_norm_stats", &nd_rasterize_backward_topk_norm_stats_tensor);

    m.def("nd_rasterize_forward_topk_norm_hexels", &nd_rasterize_forward_topk_norm_hexels_tensor);
    m.def("nd_rasterize_backward_topk_norm_hexels", &nd_rasterize_backward_topk_norm_hexels_tensor);
    m.def("nd_rasterize_forward_topk_norm_hexels_softness", &nd_rasterize_forward_topk_norm_hexels_softness_tensor);
    m.def("nd_rasterize_backward_topk_norm_hexels_softness", &nd_rasterize_backward_topk_norm_hexels_softness_tensor);
    m.def(
        "nd_rasterize_forward_topk_norm_hexels_color_grad",
        &nd_rasterize_forward_topk_norm_hexels_color_grad_tensor
    );
    m.def(
        "nd_rasterize_backward_topk_norm_hexels_color_grad",
        &nd_rasterize_backward_topk_norm_hexels_color_grad_tensor
    );
    m.def(
        "nd_rasterize_forward_topk_norm_hexels_color_grad_softness",
        &nd_rasterize_forward_topk_norm_hexels_color_grad_softness_tensor
    );
    m.def(
        "nd_rasterize_backward_topk_norm_hexels_color_grad_softness",
        &nd_rasterize_backward_topk_norm_hexels_color_grad_softness_tensor
    );
    m.def(
        "nd_rasterize_forward_topk_norm_hexels_affine",
        &nd_rasterize_forward_topk_norm_hexels_affine_tensor
    );
    m.def(
        "nd_rasterize_backward_topk_norm_hexels_affine",
        &nd_rasterize_backward_topk_norm_hexels_affine_tensor
    );
    m.def(
        "nd_rasterize_forward_topk_norm_hexels_affine_softness",
        &nd_rasterize_forward_topk_norm_hexels_affine_softness_tensor
    );
    m.def(
        "nd_rasterize_backward_topk_norm_hexels_affine_softness",
        &nd_rasterize_backward_topk_norm_hexels_affine_softness_tensor
    );
    m.def(
        "nd_rasterize_forward_topk_norm_hexels_affine_color_grad",
        &nd_rasterize_forward_topk_norm_hexels_affine_color_grad_tensor
    );
    m.def(
        "nd_rasterize_backward_topk_norm_hexels_affine_color_grad",
        &nd_rasterize_backward_topk_norm_hexels_affine_color_grad_tensor
    );
    m.def(
        "nd_rasterize_forward_topk_norm_hexels_affine_color_grad_softness",
        &nd_rasterize_forward_topk_norm_hexels_affine_color_grad_softness_tensor
    );
    m.def(
        "nd_rasterize_backward_topk_norm_hexels_affine_color_grad_softness",
        &nd_rasterize_backward_topk_norm_hexels_affine_color_grad_softness_tensor
    );

    m.def("nd_rasterize_forward_no_tiles", &nd_rasterize_forward_no_tiles_tensor);
    m.def("nd_rasterize_backward_no_tiles", &nd_rasterize_backward_no_tiles_tensor);

    m.def("nd_rasterize_forward_simple", &nd_rasterize_forward_simple_tensor);
    m.def("nd_rasterize_backward_simple", &nd_rasterize_backward_simple_tensor);

    m.def("rasterize_forward", &rasterize_forward_tensor);
    m.def("rasterize_backward", &rasterize_backward_tensor);

    m.def("project_gaussians_2d_scale_rot_forward", &project_gaussians_2d_scale_rot_forward_tensor);
    m.def("project_gaussians_2d_scale_rot_backward", &project_gaussians_2d_scale_rot_backward_tensor);

    // utils
    m.def("compute_cov2d_bounds", &compute_cov2d_bounds_tensor);
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);
}
