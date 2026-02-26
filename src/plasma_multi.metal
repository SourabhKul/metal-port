#include <metal_stdlib>
using namespace metal;

struct ParticleState {
    float3 position;
    float3 velocity;
};

// High-Fidelity Boris Integrator (Standard for Supercomputer Benchmarks like GTC)
kernel void boris_kernel(
    device ParticleState* particles [[buffer(0)]],
    constant float3& B_field [[buffer(1)]],
    constant float3& E_field [[buffer(2)]],
    constant float& dt [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float3 x = particles[id].position;
    float3 v = particles[id].velocity;
    
    float q_m = 1.0f; // charge/mass ratio
    float dt2 = dt * 0.5f;
    
    // 1. Half-step electric field push
    float3 v_minus = v + q_m * E_field * dt2;
    
    // 2. Magnetic rotation
    float3 t = q_m * B_field * dt2;
    float3 s = 2.0f * t / (1.0f + dot(t, t));
    float3 v_prime = v_minus + cross(v_minus, t);
    float3 v_plus = v_minus + cross(v_prime, s);
    
    // 3. Second half-step electric field push
    v = v_plus + q_m * E_field * dt2;
    
    // 4. Position update
    x = x + v * dt;
    
    // Save state
    particles[id].velocity = v;
    particles[id].position = x;
}

struct ShimParams {
    float3 B_offset;
    float3 E_offset;
};

struct MuonState {
    float3 position;
    float3 velocity;
    float3 spin;
};

kernel void muon_g2_kernel(
    device MuonState* muons [[buffer(0)]],
    constant float3& B_base  [[buffer(1)]],
    constant float3& E_base  [[buffer(2)]],
    constant float& dt       [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    float3 x = muons[id].position;
    float3 v = muons[id].velocity;
    float3 s = muons[id].spin;
    
    const float q_m = 1.0f; // Simplified for initial scaling test
    const float a_mu = 11659208e-10f; // Anomalous magnetic moment
    const float c = 299792458.0f;
    
    // 1. Relativistic Boris Push for Trajectory
    float v2 = dot(v, v);
    float gamma = 1.0f / sqrt(1.0f - v2 / (c * c));
    float dt2 = dt * 0.5f;
    
    // Simplified Lorentz (can be upgraded to full relativistic)
    float3 v_minus = v + q_m * E_base * dt2;
    float3 t = q_m * B_base * dt2 / gamma;
    float3 s_rot = 2.0f * t / (1.0f + dot(t, t));
    float3 v_prime = v_minus + cross(v_minus, t);
    v = v_minus + cross(v_prime, s_rot);
    x = x + v * dt;
    
    // 2. T-BMT Spin Precession (ds/dt = Omega x s)
    // Omega_a = - (q/m) * [ a_mu * B - (a_mu * gamma / (gamma + 1)) * (beta . B) * beta ]
    float3 beta = v / c;
    float3 Omega_a = - (q_m) * (a_mu * B_base - (a_mu * gamma / (gamma + 1.0f)) * dot(beta, B_base) * beta);
    
    // Rotation of spin vector using Rodrigues' formula or similar
    float3 s_new = s + cross(Omega_a * dt, s);
    s_new = normalize(s_new); // Maintain unit vector
    
    // Update
    muons[id].position = x;
    muons[id].velocity = v;
    muons[id].spin = s_new;
}

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
