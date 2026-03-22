static const uint AUX_DIM = PYTHON_AUX_DIM;

import intersect;
import utils;
import safe_math;
import camera;

struct CtrlPt: IDifferentiable {
    float4 rgba;
    float2 dist;
    float density;
    float4 xyzd;
    float aux[AUX_DIM]; // <--- Added
};

struct SplatTetrahedra_interp : IDifferentiable
{
    uint4 vtet;
    Tetrahedra tet;
    float3 base_color;
    float3 grd;
    float density;
    float aux[AUX_DIM]; // <--- Added
};

void print_verts(float4x3 verts) {
    printf("%f, %f, %f, %f\n", verts[0].x, verts[1].x, verts[2].x, verts[3].x);
}
void print_vec(float3 vec) {
    printf("(%f, %f, %f),", vec.x, vec.y, vec.z);
}

[Differentiable]
inline float phi(float x) {
    if (abs(x) < 1e-6) {
        return 1.0f - x / 2.0f;
    }
    // return (1.0f - exp(-x)) / x;
    return -(safe_expm1(-x)) / x;
}

[Differentiable]
inline float compute_integral1D(float c0, float c1, float ddt) {
    float alpha = safe_exp(-ddt);
    float phi_val = phi(ddt);
    float w0 = phi_val - alpha;
    float w1 = 1.0f - phi_val;
    return w0 * c0 + w1 * c1;
}
[Differentiable]
inline float3 compute_integral(float3 c0, float3 c1, float ddt) {
    float alpha = safe_exp(-ddt);
    float phi_val = phi(ddt);
    float w0 = phi_val - alpha;
    float w1 = 1.0f - phi_val;
    return w0 * c0 + w1 * c1;
}

[Differentiable]
inline CtrlPt integrate_tetra_segment(
    in SplatTetrahedra_interp g, 
    in Ray ray,
    in float enter,
    in float exit,
    in float3 enter_norm,
    in float3 exit_norm,
)
{
    float dt = max(exit - enter, 0.0f);

    float3 v_enter = enter * ray.d + ray.o - g.tet.verts[0];
    float3 v_exit = exit * ray.d + ray.o - g.tet.verts[0];
    float3 c_enter = max(dot(g.grd, v_enter) + g.base_color, 0.f);
    float3 c_exit = max(dot(g.grd, v_exit) + g.base_color, 0.f);
    // float3 c_enter = dot(g.grd, v_enter) + g.base_color;
    // float3 c_exit = dot(g.grd, v_exit) + g.base_color;

    // float alpha = safe_clip(1.0f - safe_exp(-g.density * dt), 0.f, 1.f);
    float alpha = max(-safe_expm1(-g.density * dt), 0.f);
    float3 final_color = compute_integral(c_exit, c_enter, dt * g.density);

    float3 rendered_norm = alpha * l2_normalize(enter_norm);
    float depth = safe_div(compute_integral1D(enter, exit, dt * g.density), ray.ortho);

    float4 rgba = { final_color.x,
                    final_color.y,
                    final_color.z,
                    g.density * dt };

    CtrlPt result;
    result.rgba = { final_color.x, final_color.y, final_color.z, g.density * dt };
    result.dist = { enter, exit };
    result.density = g.density;
    result.xyzd = float4(rendered_norm, depth);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        result.aux[i] = g.aux[i] * detach(alpha);
    }
    
    return result;
}

[Differentiable]
inline bool evaluate_tetra_interp(
    in SplatTetrahedra_interp g, 
    in Ray ray,
    no_diff in float t_min,
    out CtrlPt ctrl_pt)
{
    float2 dist;

    float4x3 verts = float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

    float3 enter_norm, exit_norm;
    bool hit = ray_tetrahedron_intersect_fused(
        ray.o, ray.d, verts, dist, enter_norm, exit_norm);

    float enter = dist.x;
    float exit = max(dist.y, enter);

    if (!hit) {
        ctrl_pt = { (float4)(0.f, 0.f, 0.f, 0.f), {0.f, 0.f}, g.density };
        return false;
    } else {
        ctrl_pt = integrate_tetra_segment(g, ray, enter, exit, enter_norm, exit_norm);
        return true;
    }
}

[Differentiable]
inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView cell_values)
{
    no_diff uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(cell_values[g_idx, 0]), 0.f);
    float3 base_color = {
        float(cell_values[g_idx, 1]),
        float(cell_values[g_idx, 2]),
        float(cell_values[g_idx, 3]),
    };
    float3 grd = {
        float(cell_values[g_idx, 4]),
        float(cell_values[g_idx, 5]),
        float(cell_values[g_idx, 6]),
    };

    SplatTetrahedra_interp result = {
        virtual_tet, tet, base_color, grd, density
    };

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        // 1 (density) + 3 (color) + 3 (grd) = 7
        result.aux[i] = float(cell_values[g_idx, 7 + i]);
    }

    return result;
}

inline SplatTetrahedra_interp load_tet_alphablend_interp(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  cell_values)
{
    uint4 virtual_tet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet    = load_tetrahedra(vertices, virtual_tet);

    float density = max(float(cell_values[g_idx, 0]), 0.f);
    SplatTetrahedra_interp result;
    result.vtet = virtual_tet;
    result.tet       = tet;
    result.density   = density;
    result.base_color = {
        float(cell_values[g_idx, 1]),
        float(cell_values[g_idx, 2]),
        float(cell_values[g_idx, 3]),
    };
    result.grd = {
        float(cell_values[g_idx, 4]),
        float(cell_values[g_idx, 5]),
        float(cell_values[g_idx, 6]),
    };
    for (int i = 0; i < AUX_DIM; ++i) {
        // 1 (density) + 3 (color) + 3 (grd) = 7
        result.aux[i] = float(cell_values[g_idx, 7 + i]);
    }
    return result;
}

// ===========================================================================
// Per-vertex attribute model with N-sample quadrature
// ===========================================================================
static const int N_QUAD_SAMPLES = PYTHON_N_QUAD_SAMPLES;

struct SplatTetrahedra_vertex : IDifferentiable
{
    uint4 vtet;
    Tetrahedra tet;
    float4 v_sigma_rgb[4];  // per-vertex (sigma, r, g, b)
};

[Differentiable]
inline SplatTetrahedra_vertex load_tet_vertex(
    int32_t        g_idx,
    DiffTensorView vertices,
    TensorView<int32_t> indices,
    DiffTensorView vertex_values)
{
    no_diff uint4 vtet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, vtet);

    SplatTetrahedra_vertex r;
    r.vtet = vtet;
    r.tet = tet;
    r.v_sigma_rgb[0] = read_t4_float4(vtet.x, vertex_values);
    r.v_sigma_rgb[1] = read_t4_float4(vtet.y, vertex_values);
    r.v_sigma_rgb[2] = read_t4_float4(vtet.z, vertex_values);
    r.v_sigma_rgb[3] = read_t4_float4(vtet.w, vertex_values);
    return r;
}

inline SplatTetrahedra_vertex load_tet_vertex(
    int32_t        g_idx,
    TensorView<float> vertices,
    TensorView<int32_t> indices,
    TensorView<float>  vertex_values)
{
    uint4 vtet = load_virtual_tetrahedra(g_idx, indices);
    Tetrahedra tet = load_tetrahedra(vertices, vtet);

    SplatTetrahedra_vertex r;
    r.vtet = vtet;
    r.tet = tet;
    r.v_sigma_rgb[0] = read_t4_float4(vtet.x, vertex_values);
    r.v_sigma_rgb[1] = read_t4_float4(vtet.y, vertex_values);
    r.v_sigma_rgb[2] = read_t4_float4(vtet.z, vertex_values);
    r.v_sigma_rgb[3] = read_t4_float4(vtet.w, vertex_values);
    return r;
}

[Differentiable]
inline CtrlPt integrate_tetra_segment_vertex(
    in SplatTetrahedra_vertex g,
    in Ray ray,
    in float enter,
    in float exit,
    in float3 enter_norm,
    in float3 exit_norm)
{
    float dt = max(exit - enter, 0.0f);

    // Cramer's rule for M_inv: maps world offset to bary coords (1,2,3)
    float3 e1 = g.tet.verts[1] - g.tet.verts[0];
    float3 e2 = g.tet.verts[2] - g.tet.verts[0];
    float3 e3 = g.tet.verts[3] - g.tet.verts[0];
    float det_val = dot(e1, cross(e2, e3));

    // Safe inverse: for degenerate tets, ray-tet intersection already
    // rejects them, so this just prevents NaN in autodiff
    float inv_det = safe_div(1.0f, det_val);
    float3 r0 = cross(e2, e3) * inv_det;
    float3 r1 = cross(e3, e1) * inv_det;
    float3 r2 = cross(e1, e2) * inv_det;

    // N-sample quadrature: samples at midpoints of N equal sub-segments
    float sub_dt = dt / float(N_QUAD_SAMPLES);
    float3 rgb = float3(0.f);
    float total_tau = 0.f;
    float T_accum = 1.0f;
    float sigma_sum = 0.f;

    [ForceUnroll]
    for (int s = 0; s < N_QUAD_SAMPLES; s++) {
        float t_s = enter + (float(s) + 0.5f) * sub_dt;
        float3 p_s = t_s * ray.d + ray.o - g.tet.verts[0];

        float3 bary_123 = float3(dot(r0, p_s), dot(r1, p_s), dot(r2, p_s));
        float bary_0 = 1.0f - bary_123.x - bary_123.y - bary_123.z;

        float4 attr_s = bary_0 * g.v_sigma_rgb[0] + bary_123.x * g.v_sigma_rgb[1]
                       + bary_123.y * g.v_sigma_rgb[2] + bary_123.z * g.v_sigma_rgb[3];

        float sigma_s = safe_exp(attr_s.x);
        float3 color_s = float3(attr_s.y, attr_s.z, attr_s.w);

        float tau_s = sigma_s * sub_dt;
        float alpha_s = max(-safe_expm1(-tau_s), 0.f);
        rgb = rgb + T_accum * alpha_s * color_s;
        T_accum = T_accum * safe_exp(-tau_s);
        total_tau = total_tau + tau_s;
        sigma_sum = sigma_sum + sigma_s;
    }

    float avg_sigma = sigma_sum / float(N_QUAD_SAMPLES);
    float alpha_total = max(-safe_expm1(-total_tau), 0.f);
    float3 rendered_norm = alpha_total * l2_normalize(enter_norm);
    float depth = safe_div(compute_integral1D(enter, exit, total_tau), ray.ortho);

    CtrlPt result;
    result.rgba = float4(rgb, total_tau);
    result.dist = float2(enter, exit);
    result.density = avg_sigma;
    result.xyzd = float4(rendered_norm, depth);

    [ForceUnroll]
    for (int i = 0; i < AUX_DIM; ++i) {
        result.aux[i] = 0.f;
    }

    return result;
}

[Differentiable]
inline bool evaluate_tetra_vertex(
    in SplatTetrahedra_vertex g,
    in Ray ray,
    no_diff in float t_min,
    out CtrlPt ctrl_pt)
{
    float2 dist;
    float4x3 verts = float4x3(g.tet.verts[0], g.tet.verts[1], g.tet.verts[2], g.tet.verts[3]);

    float3 enter_norm, exit_norm;
    bool hit = ray_tetrahedron_intersect_fused(
        ray.o, ray.d, verts, dist, enter_norm, exit_norm);

    float enter = dist.x;
    float exit = max(dist.y, enter);

    if (!hit) {
        ctrl_pt.rgba = float4(0.f);
        ctrl_pt.dist = float2(0.f);
        ctrl_pt.density = 0.f;
        ctrl_pt.xyzd = float4(0.f);
        [ForceUnroll] for (int i = 0; i < AUX_DIM; ++i) ctrl_pt.aux[i] = 0.f;
        return false;
    } else {
        ctrl_pt = integrate_tetra_segment_vertex(g, ray, enter, exit, enter_norm, exit_norm);
        return true;
    }
}
