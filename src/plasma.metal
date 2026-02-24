#include <metal_stdlib>
using namespace metal;

struct ParticleState {
    float3 position;
    float3 velocity;
};

kernel void plasma_kernel(
    device ParticleState* particles [[buffer(0)]],
    constant float3& B_field [[buffer(1)]],
    constant float3& E_field [[buffer(2)]],
    constant float& dt [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float3 pos = particles[id].position;
    float3 vel = particles[id].velocity;
    
    // Lorentz Force: F = q(E + v x B)
    // For simplicity, we assume q/m = 1.0 in this phase
    float3 v_cross_B = cross(vel, B_field);
    float3 accel = E_field + v_cross_B;
    
    // Update
    particles[id].velocity = vel + accel * dt;
    particles[id].position = pos + particles[id].velocity * dt;
}
