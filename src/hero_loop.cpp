#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

#define PARTICLE_COUNT 1000000
#define TIMESTEPS 1200000

int main() {
    // Metal Device Setup
    auto device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal Support Unavailable" << std::endl;
        return -1;
    }
    
    std::cout << "Device: " << device->name()->utf8String() << std::endl;

    // Allocate Buffers (+1 GB safety margin for struct constants)
    auto buffer = device->newBuffer(PARTICLE_COUNT * sizeof(float)*5, MTL::ResourceStorageModeShared);
    }