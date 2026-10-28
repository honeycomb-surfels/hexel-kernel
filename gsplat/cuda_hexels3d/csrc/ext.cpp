#include "hexels3d_forward.h"
#include "hexels3d_backward.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_hexels3d_forward_cached", &rasterize_hexels3d_forward_cached);
    m.def("rasterize_hexels3d_backward_cached", &rasterize_hexels3d_backward_cached);
    m.def("rasterize_hexels3d_forward", &rasterize_hexels3d_forward);
    m.def("rasterize_hexels3d_owner_forward", &rasterize_hexels3d_owner_forward);
    m.def("rasterize_hexels3d_forward_with_contrib", &rasterize_hexels3d_forward_with_contrib);
    m.def("rasterize_hexels3d_forward_with_overdraw", &rasterize_hexels3d_forward_with_overdraw);
    m.def("rasterize_hexels3d_forward_with_contrib_overdraw", &rasterize_hexels3d_forward_with_contrib_overdraw);
    m.def("rasterize_hexels3d_forward_with_overdraw_hits", &rasterize_hexels3d_forward_with_overdraw_hits);
    m.def(
        "rasterize_hexels3d_forward_with_contrib_overdraw_hits",
        &rasterize_hexels3d_forward_with_contrib_overdraw_hits
    );
    m.def("rasterize_hexels3d_forward_all", &rasterize_hexels3d_forward_all);
    m.def("rasterize_hexels3d_forward_median", &rasterize_hexels3d_forward_median);
    m.def("rasterize_hexels3d_exact_sort_forward", &rasterize_hexels3d_exact_sort_forward);
    m.def("rasterize_hexels3d_pixel_risk_forward", &rasterize_hexels3d_pixel_risk_forward);
    m.def("rasterize_hexels3d_backward", &rasterize_hexels3d_backward);
}
