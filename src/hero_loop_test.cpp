// Removed Foundation headers for pure Metal-only workflow
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>
#include <iomanip>

#define TEST_PARTICLE_COUNT 10000
#define TEST_TIMESTEPS 100

// Simulating a small run for testing isolation and compatibility with active workloads
int main() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    auto device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal support unavailable." << std::endl;
        return -1;
    }

    std::cout << "Device: " << device->name()->utf8String() << std::endl;
    auto buffer = device->newBuffer(TEST_PARTICLE_COUNT * sizeof(float) * 6, MTL::ResourceStorageModeShared);
    float* bufferPtr = static_cast<float*>(buffer->contents());
    for (int i = 0; i < TEST_PARTICLE_COUNT * 6; i++) {
        bufferPtr[i] = static_cast<float>(i % TEST_PARTICLE_COUNT) / TEST_PARTICLE_COUNT;
    }

    // Kernel prep and small launch omitted for brevity

    std::cout << "Test run complete." << std::endl;
    buffer->release();
    pool->release();
    return 0;
}