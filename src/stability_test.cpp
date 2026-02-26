#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

struct ParticleState {
    float pos[4]; // float3 is 16-byte aligned in Metal
    float vel[4];
};

int main() {
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();

    const uint32_t numParticles = 1024;
    const uint32_t numSteps = 100000;
    const float dt = 0.01f; // Larger dt to see drift more clearly

    std::cout << "--- Metal C++: Integrator Stability & Energy Conservation Test ---" << std::endl;
    std::cout << "Steps: " << numSteps << " | dt: " << dt << " | Precision: FP32" << std::endl;

    // Load Kernel
    FILE* pFile = fopen("./src/plasma_multi.metal", "rb");
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

    MTL::Function* pFunc = pLibrary->newFunction(NS::String::string("boris_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPSO = pDevice->newComputePipelineState(pFunc, &pError);

    // Buffers
    MTL::Buffer* pStateBuffer = pDevice->newBuffer(numParticles * sizeof(ParticleState), MTL::ResourceStorageModeShared);
    float B_field[4] = {0.0f, 0.0f, 1.0f, 0.0f}; // float3 padding
    float E_field[4] = {0.0f, 0.0f, 0.0f, 0.0f}; 
    MTL::Buffer* pBBuffer = pDevice->newBuffer(sizeof(float)*4, MTL::ResourceStorageModeShared);
    MTL::Buffer* pEBuffer = pDevice->newBuffer(sizeof(float)*4, MTL::ResourceStorageModeShared);
    MTL::Buffer* pDtBuffer = pDevice->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    memcpy(pBBuffer->contents(), B_field, sizeof(float)*4);
    memcpy(pEBuffer->contents(), E_field, sizeof(float)*4);
    memcpy(pDtBuffer->contents(), &dt, sizeof(float));

    // Init: All particles with vel=(1,0,0) -> Energy = 0.5
    ParticleState* pParticles = static_cast<ParticleState*>(pStateBuffer->contents());
    for(uint32_t i=0; i<numParticles; ++i) {
        pParticles[i].pos[0] = 0.0f; pParticles[i].pos[1] = 0.0f; pParticles[i].pos[2] = 0.0f; pParticles[i].pos[3] = 0.0f;
        pParticles[i].vel[0] = 1.0f; pParticles[i].vel[1] = 0.0f; pParticles[i].vel[2] = 0.0f; pParticles[i].vel[3] = 0.0f;
    }

    auto get_energy = [&]() {
        double total = 0;
        for(uint32_t i=0; i<numParticles; ++i) {
            float vx = pParticles[i].vel[0];
            float vy = pParticles[i].vel[1];
            float vz = pParticles[i].vel[2];
            total += 0.5 * (vx*vx + vy*vy + vz*vz);
        }
        return total / numParticles;
    };

    double E0 = get_energy();
    std::cout << "Initial Energy: " << std::fixed << std::setprecision(10) << E0 << std::endl;

    MTL::CommandQueue* pQueue = pDevice->newCommandQueue();

    for (uint32_t step = 0; step < numSteps; ++step) {
        NS::AutoreleasePool* pLoopPool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* pCmdBuf = pQueue->commandBuffer();
        MTL::ComputeCommandEncoder* pEncoder = pCmdBuf->computeCommandEncoder();
        pEncoder->setComputePipelineState(pPSO);
        pEncoder->setBuffer(pStateBuffer, 0, 0);
        pEncoder->setBuffer(pBBuffer, 0, 1);
        pEncoder->setBuffer(pEBuffer, 0, 2);
        pEncoder->setBuffer(pDtBuffer, 0, 3);
        pEncoder->dispatchThreads(MTL::Size::Make(numParticles, 1, 1), MTL::Size::Make(256, 1, 1));
        pEncoder->endEncoding();
        pCmdBuf->commit();
        
        if (step % 10000 == 0 && step > 0) {
            pCmdBuf->waitUntilCompleted();
            double E = get_energy();
            double drift = (E - E0) / E0;
            std::cout << "[Step " << step << "] Rel. Energy Error: " << std::scientific << drift << std::endl;
        }
        pLoopPool->release();
    }

    MTL::CommandBuffer* pFinalSync = pQueue->commandBuffer();
    pFinalSync->commit();
    pFinalSync->waitUntilCompleted();

    double Ef = get_energy();
    std::cout << "Final Energy: " << std::fixed << std::setprecision(10) << Ef << std::endl;
    std::cout << "Total Relative Drift: " << std::scientific << (Ef - E0)/E0 << std::endl;

    return 0;
}
