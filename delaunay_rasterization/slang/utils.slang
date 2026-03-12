// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import intersect;
import safe_math;

static const float FLT_MAX = 1e20;
static const float FLT_MIN = -1e20;
static const float eps = 1e-7;

[Differentiable]
float read_t1_float(uint32_t idx, DiffTensorView t1)
{
    return t1[uint2(idx, 0)];
}

float3 read_t3_float3(uint32_t idx, TensorView<float> t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float3 read_t3_float3(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable]
float4 read_t4_float4(uint32_t idx, DiffTensorView t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

float4 read_t4_float4(uint32_t idx, TensorView<float> t4)
{
    return float4(t4[uint2(idx, 0)],
                  t4[uint2(idx, 1)],
                  t4[uint2(idx, 2)],
                  t4[uint2(idx, 3)]);
}

[Differentiable]
float4 read_t2x2_float4(uint32_t idx, DiffTensorView t22)
{
    return float4(t22[uint3(idx, 0, 0)],
                  t22[uint3(idx, 1, 0)],
                  t22[uint3(idx, 0, 1)],
                  t22[uint3(idx, 1, 1)]);
}

[Differentiable]
float2x2 read_t2x2_float2x2(uint32_t idx, DiffTensorView t2x2)
{
    return float2x2(t2x2[uint3(idx, 0, 0)],
                    t2x2[uint3(idx, 1, 0)],
                    t2x2[uint3(idx, 0, 1)],
                    t2x2[uint3(idx, 1, 1)]);
}


float clip(float val, float min_val, float max_val)
{
    return max(min_val, min(max_val, val));
}


uint4 load_virtual_tetrahedra(int32_t g_idx,
                              TensorView<int32_t> indices)
{
    return { 
        indices[uint2(g_idx, 0)],
        indices[uint2(g_idx, 1)],
        indices[uint2(g_idx, 2)],
        indices[uint2(g_idx, 3)]
    };
}

[Differentiable]
Tetrahedra load_tetrahedra(DiffTensorView vertices,
                           uint4 virtual_tetra)
{
    float3[4] tetra = {
        read_t3_float3(virtual_tetra.x, vertices),
        read_t3_float3(virtual_tetra.y, vertices),
        read_t3_float3(virtual_tetra.z, vertices),
        read_t3_float3(virtual_tetra.w, vertices),
    };
    return { tetra };
}

Tetrahedra load_tetrahedra(TensorView<float> vertices,
                           uint4 virtual_tetra)
{
    float3[4] tetra = {
        read_t3_float3(virtual_tetra.x, vertices),
        read_t3_float3(virtual_tetra.y, vertices),
        read_t3_float3(virtual_tetra.z, vertices),
        read_t3_float3(virtual_tetra.w, vertices),
    };
    return { tetra };
}