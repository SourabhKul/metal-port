#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <vector>

struct ParticleState {
    float pos[3];
    float vel[3];
};

struct ShimParams {
    float B_offset[3];
    float E_offset[3];
};

int main(int argc, char* argv[]) {
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();
    
    // Default parameters for SBI
    uint32_t numParticlesPerUniverse = 100000;
    uint32_t numUniverses = 100;
    uint32_t numSteps = 10000; 
    float dt = 0.001f;
    bool isBenchmarkMode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--benchmark-summit") {
            isBenchmarkMode = true;
            numParticlesPerUniverse = 100000000; // 100M particles
            numUniverses = 1;
            numSteps = 10000; // 1 Trillion total updates (100M * 10k)
        } else if (i == 1) {
            numUniverses = std::stoi(arg);
        }
    }

    if (isBenchmarkMode) {
        std::cout << "--- Metal C++ SBI: Supercomputer Benchmark Mode (GTC/Summit) ---" << std::endl;
        std::cout << "Workload: 1 Trillion Total Particle Pushes (Boris Integrator)" << std::endl;
    } else {
        std::cout << "--- Metal C++ SBI: Multi-Universe Mode ---" << std::endl;
    }
    std::cout << "Universes: " << numUniverses << " | Particles/Universe: " << numParticlesPerUniverse << std::endl;

    // Load Multi-Kernel Shader Source
    FILE* pFile = fopen("./src/plasma_multi.metal", "rb");
    if (!pFile) {
        std::cerr << "Error: Could not open plasma_multi.metal" << std::endl;
        return -1;
    }
    fseek(pFile, 0, SEEK_END);
    long length = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);
    char* pSource = (char*)malloc(length + 1);
    fread(pSource, 1, length, pFile);
    pSource[length] = '\0';
    fclose(pFile);

    NS::Error* pError = nullptr;
    MTL::CompileOptions* pOptions = MTL::CompileOptions::alloc()->init();
    NS::String* pSourceString = NS::String::string(pSource, NS::UTF8StringEncoding);
    MTL::Library* pLibrary = pDevice->newLibrary(pSourceString, pOptions, &pError);
    free(pSource);

    if (!pLibrary) {
        std::cerr << "Error: Failed to compile Metal library." << std::endl;
        if (pError) std::cerr << pError->localizedDescription()->utf8String() << std::endl;
        return -1;
    }

    MTL::Function* pFunc = pLibrary->newFunction(NS::String::string(isBenchmarkMode ? "boris_kernel" : "plasma_kernel_multi", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPSO = pDevice->newComputePipelineState(pFunc, &pError);

    if (!pPSO) {
        std::cerr << "Error: Failed to create PSO." << std::endl;
        return -1;
    }

    // Allocate Buffers
    size_t stateSize = numUniverses * numParticlesPerUniverse * sizeof(ParticleState);
    MTL::Buffer* pStateBuffer = pDevice->newBuffer(stateSize, MTL::ResourceStorageModeShared);
    MTL::Buffer* pShimBuffer = pDevice->newBuffer(numUniverses * sizeof(ShimParams), MTL::ResourceStorageModeShared);
    
    float B_base[3] = {0.0f, 0.0f, 1.0f};
    float E_base[3] = {0.1f, 0.0f, 0.0f};
    MTL::Buffer* pBBuffer = pDevice->newBuffer(sizeof(B_base), MTL::ResourceStorageModeShared);
    MTL::Buffer* pEBuffer = pDevice->newBuffer(sizeof(E_base), MTL::ResourceStorageModeShared);
    MTL::Buffer* pDtBuffer = pDevice->newBuffer(sizeof(dt), MTL::ResourceStorageModeShared);

    memcpy(pBBuffer->contents(), B_base, sizeof(B_base));
    memcpy(pEBuffer->contents(), E_base, sizeof(E_base));
    memcpy(pDtBuffer->contents(), &dt, sizeof(dt));

    // Initialize Particles and Shims
    ParticleState* pParticles = static_cast<ParticleState*>(pStateBuffer->contents());
    for(uint32_t i=0; i<numUniverses * numParticlesPerUniverse; ++i) {
        pParticles[i].pos[0] = 0.0f; pParticles[i].pos[1] = 0.0f; pParticles[i].pos[2] = 0.0f;
        pParticles[i].vel[0] = 0.0f; pParticles[i].vel[1] = 1.0f; pParticles[i].vel[2] = 0.0f;
    }

    ShimParams* pShims = static_cast<ShimParams*>(pShimBuffer->contents());
    for(uint32_t i=0; i<numUniverses; ++i) {
        for(int j=0; j<3; ++j) {
            pShims[i].B_offset[j] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
            pShims[i].E_offset[j] = 0.0f;
        }
    }

    MTL::CommandQueue* pQueue = pDevice->newCommandQueue();
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting ABC-SMC generation: " << numUniverses << " universes..." << std::endl;

    for (uint32_t step = 0; step < numSteps; ++step) {
        NS::AutoreleasePool* pLoopPool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* pCmdBuf = pQueue->commandBuffer();
        MTL::ComputeCommandEncoder* pEncoder = pCmdBuf->computeCommandEncoder();
        
        pEncoder->setComputePipelineState(pPSO);
        pEncoder->setBuffer(pStateBuffer, 0, 0);
        if (!isBenchmarkMode) {
            pEncoder->setBuffer(pShimBuffer, 0, 1);
            pEncoder->setBuffer(pBBuffer, 0, 2);
            pEncoder->setBuffer(pEBuffer, 0, 3);
            pEncoder->setBuffer(pDtBuffer, 0, 4);
        } else {
            pEncoder->setBuffer(pBBuffer, 0, 1);
            pEncoder->setBuffer(pEBuffer, 0, 2);
            pEncoder->setBuffer(pDtBuffer, 0, 3);
        }
        
        MTL::Size gridSize = MTL::Size::Make(numUniverses * numParticlesPerUniverse, 1, 1);
        MTL::Size threadgroupSize = MTL::Size::Make(pPSO->maxTotalThreadsPerThreadgroup(), 1, 1);
        
        pEncoder->dispatchThreads(gridSize, threadgroupSize);
        pEncoder->endEncoding();
        pCmdBuf->commit();
        
        // Accurate Timing: Wait every step in benchmark mode to prevent submission scaling
        if (isBenchmarkMode) {
            pCmdBuf->waitUntilCompleted();
            if (step % 10 == 0) {
                 auto now = std::chrono::high_resolution_clock::now();
                 std::chrono::duration<double> d = now - start;
                 double rate = (double)numParticlesPerUniverse * (step+1) / d.count();
                 std::cout << "[Step " << step << "/" << numSteps << "] | Rate: " << rate / 1e9 << " B pt-up/s" << std::endl;
            }
        } else if (step % 100 == 0 && step > 0) {
            pCmdBuf->waitUntilCompleted();
        }
        
        pLoopPool->release();
    }
    
    // Final sync
    if (!isBenchmarkMode) {
        MTL::CommandBuffer* pFinalSync = pQueue->commandBuffer();
        pFinalSync->commit();
        pFinalSync->waitUntilCompleted();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "ABC-SMC Batch Complete in " << diff.count() << "s" << std::endl;

    // Cleanup
    pPSO->release(); pFunc->release(); pLibrary->release(); pOptions->release();
    pStateBuffer->release(); pShimBuffer->release(); pBBuffer->release(); pEBuffer->release(); pDtBuffer->release();
    pQueue->release(); pDevice->release(); pPool->release();
    return 0;
}
