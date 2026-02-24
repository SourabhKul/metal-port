#include <metal_stdlib>
using namespace metal;

struct ParticleState {
    float3 position;
    float3 velocity;
};

struct ShimParams {
    float3 B_offset;
    float3 E_offset;
};

kernel void plasma_kernel_multi(
    device ParticleState* particles [[buffer(0)]],
    constant ShimParams* shims      [[buffer(1)]],
    constant float3& B_base         [[buffer(2)]],
    constant float3& E_base         [[buffer(3)]],
    constant float& dt              [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]])
{
    // Each threadgroup represents one 'universe' (one shim configuration)
    // For simplicity, we assume particles are partitioned by threadgroup size
    uint shim_idx = group_id; 
    
    float3 pos = particles[id].position;
    float3 vel = particles[id].velocity;
    
    float3 B_total = B_base + shims[shim_idx].B_offset;
    float3 E_total = E_base + shims[shim_idx].E_offset;
    
    // Lorentz Force
    float3 v_cross_B = cross(vel, B_total);
    float3 accel = E_total + v_cross_B;
    
    // Update
    particles[id].velocity = vel + accel * dt;
    particles[id].position = pos + particles[id].velocity * dt;
}
