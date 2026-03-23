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

    float inv_det = safe_div(1.0f, det_val);
    float3 r0 = cross(e2, e3) * inv_det;
    float3 r1 = cross(e3, e1) * inv_det;
    float3 r2 = cross(e1, e2) * inv_det;

    // Evaluate attributes at entry and exit points
    float3 p_enter = enter * ray.d + ray.o - g.tet.verts[0];
    float3 p_exit  = exit  * ray.d + ray.o - g.tet.verts[0];

    float3 bary_enter = float3(dot(r0, p_enter), dot(r1, p_enter), dot(r2, p_enter));
    float b0_enter = 1.0f - bary_enter.x - bary_enter.y - bary_enter.z;
    float4 attr0 = b0_enter * g.v_sigma_rgb[0] + bary_enter.x * g.v_sigma_rgb[1]
                  + bary_enter.y * g.v_sigma_rgb[2] + bary_enter.z * g.v_sigma_rgb[3];

    float3 bary_exit = float3(dot(r0, p_exit), dot(r1, p_exit), dot(r2, p_exit));
    float b0_exit = 1.0f - bary_exit.x - bary_exit.y - bary_exit.z;
    float4 attr1 = b0_exit * g.v_sigma_rgb[0] + bary_exit.x * g.v_sigma_rgb[1]
                  + bary_exit.y * g.v_sigma_rgb[2] + bary_exit.z * g.v_sigma_rgb[3];

    // Log-densities at entry and exit (attr.x is log-density)
    float d0 = attr0.x;
    float d1 = attr1.x;
    float3 c0 = float3(attr0.y, attr0.z, attr0.w);
    float3 c1 = float3(attr1.y, attr1.z, attr1.w);

    // ── IBP-Simpson integration ──
    // Closed-form optical depth for log-linear density:
    //   σ(s) = exp(d0 + s*(d1-d0)/dt)
    //   τ = dt*e^d0*(e^δ - 1)/δ    where δ = d1-d0
    //
    // For small δ, use Taylor: (e^δ-1)/δ = 1 + δ/2 + δ²/6 + δ³/24
    // This avoids catastrophic cancellation in (e^δ - 1)/δ.
    float delta = d1 - d0;
    // Clamp log-densities to prevent overflow in exp
    float d0_c = clamp(d0, -20.f, 20.f);
    float d1_c = clamp(d1, -20.f, 20.f);
    float delta_c = d1_c - d0_c;
    float ed0 = safe_exp(d0_c);

    // (e^δ - 1)/δ with Taylor expansion for small δ
    float expm1_over_delta;
    float ed;  // e^δ
    if (abs(delta_c) < 0.01f) {
        // Taylor: 1 + δ/2 + δ²/6 + δ³/24
        float d = delta_c;
        expm1_over_delta = 1.0f + d * (0.5f + d * (1.0f/6.0f + d * (1.0f/24.0f)));
        ed = 1.0f + d * expm1_over_delta;  // e^δ ≈ 1 + δ*(e^δ-1)/δ
    } else {
        ed = safe_exp(delta_c);
        expm1_over_delta = (ed - 1.0f) / delta_c;
    }

    float tau_total = max(dt * ed0 * expm1_over_delta, 0.f);
    float exp_neg_tau = safe_exp(-tau_total);
    float alpha = max(1.0f - exp_neg_tau, 0.f);

    // Simpson's rule for I = ∫₀^dt T(s) ds using T(0), T(dt/2), T(dt)
    // τ(dt/2) = τ_total / (e^{δ/2} + 1), and e^{δ/2} = √(e^δ)
    float tau_mid;
    if (abs(delta_c) < 0.01f) {
        tau_mid = tau_total * 0.5f;
    } else {
        float ed_half = sqrt(max(ed, 0.f));  // e^{δ/2}
        tau_mid = tau_total / (ed_half + 1.0f);
    }
    float T_mid = safe_exp(-tau_mid);
    float T_end = 1.0f - alpha;

    // I/dt = (T(0) + 4*T(dt/2) + T(dt)) / 6 = (1 + 4*T_mid + T_end) / 6
    float I_over_dt = (1.0f + 4.0f * T_mid + T_end) / 6.0f;

    // IBP decomposition: C = c₀·α + (c₁-c₀)·[I/dt - (1-α)]
    float weight = I_over_dt - T_end;
    float3 rgb = c0 * alpha + (c1 - c0) * weight;

    float ed1 = ed0 * ed;
    float avg_sigma = (ed0 + ed1) * 0.5f;
    float3 rendered_norm = alpha * l2_normalize(enter_norm);
    float depth = safe_div(compute_integral1D(enter, exit, tau_total), ray.ortho);

    CtrlPt result;
    result.rgba = float4(rgb, tau_total);
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
