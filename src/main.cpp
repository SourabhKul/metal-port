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

int main() {
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    
    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();
    if (!pDevice) {
        std::cerr << "Error: Metal is not supported on this device." << std::endl;
        return -1;
    }

    std::cout << "--- Metal C++ Port: Phase 1 ---" << std::endl;
    std::cout << "Device Name: " << pDevice->name()->utf8String() << std::endl;
    std::cout << "Unified Memory: " << (pDevice->hasUnifiedMemory() ? "Yes" : "No") << std::endl;
    std::cout << "Max Buffer Size: " << pDevice->maxBufferLength() / (1024ULL*1024*1024) << " GB" << std::endl;
    
    // Unified Memory Pool Setup (128GB Context)
    // We target the max allowed or 100GB, whichever is smaller.
    size_t maxAllowed = pDevice->maxBufferLength();
    size_t targetSize = std::min((unsigned long long)maxAllowed, 100ULL * 1024 * 1024 * 1024);
    
    std::cout << "Allocating " << (double)targetSize / (1024.0*1024*1024) << " GB unified memory buffer..." << std::endl;
    
    MTL::Buffer* pStateBuffer = pDevice->newBuffer(targetSize, MTL::ResourceStorageModeShared);
    
    if (!pStateBuffer) {
        std::cerr << "Error: Failed to allocate " << (double)targetSize / (1024.0*1024*1024) << " GB buffer." << std::endl;
        pDevice->release();
        pPool->release();
        return -1;
    }
    
    void* pContents = pStateBuffer->contents();
    std::cout << "Successfully mapped buffer at physical address: " << pContents << std::endl;
    
    // Quick test: Initialize the first and last few elements
    float* pData = static_cast<float*>(pContents);
    size_t numElements = targetSize / sizeof(float);
    
    pData[0] = 1.0f;
    pData[numElements - 1] = 3.14159f;
    
    std::cout << "Memory verify: [0] = " << pData[0] << ", [" << numElements - 1 << "] = " << pData[numElements - 1] << std::endl;
    std::cout << "Phase 1: Environment and Unified Memory Mapping COMPLETE." << std::endl;

    std::cout << "\n--- Metal C++ Port: Phase 2 ---" << std::endl;
    std::cout << "Initializing Compute Pipeline..." << std::endl;

    NS::Error* pError = nullptr;
    
    // Read shader source from file
    FILE* pFile = fopen("./src/plasma.metal", "rb");
    if (!pFile) {
        std::cerr << "Error: Could not open plasma.metal" << std::endl;
        return -1;
    }
    fseek(pFile, 0, SEEK_END);
    long length = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);
    char* pSource = (char*)malloc(length + 1);
    fread(pSource, 1, length, pFile);
    pSource[length] = '\0';
    fclose(pFile);

    NS::String* pSourceString = NS::String::string(pSource, NS::UTF8StringEncoding);
    MTL::CompileOptions* pOptions = MTL::CompileOptions::alloc()->init();
    MTL::Library* pLibrary = pDevice->newLibrary(pSourceString, pOptions, &pError);
    free(pSource);

    if (!pLibrary) {
        std::cerr << "Error: Failed to compile Metal library from source." << std::endl;
        if (pError) {
            std::cerr << "Metal Error: " << pError->localizedDescription()->utf8String() << std::endl;
        }
    } else {
        NS::String* pFuncName = NS::String::string("plasma_kernel", NS::UTF8StringEncoding);
        MTL::Function* pFunc = pLibrary->newFunction(pFuncName);
        MTL::ComputePipelineState* pPSO = pDevice->newComputePipelineState(pFunc, &pError);
        
        if (pPSO) {
            std::cout << "Compute Pipeline State: CREATED (Runtime Compiled)." << std::endl;
            std::cout << "Max Total Threads Per Threadgroup: " << pPSO->maxTotalThreadsPerThreadgroup() << std::endl;
            std::cout << "Phase 2: Compute Kernel Integration COMPLETE." << std::endl;

            std::cout << "\n--- Metal C++ Port: Phase 3 (Hero Loop) ---" << std::endl;
            
            // Simulation Parameters
            const uint32_t numParticles = 1000000;
            const uint32_t numSteps = 1200000;
            const float dt = 0.001f;
            
            struct ParticleState {
                float pos[3];
                float vel[3];
            };

            // Re-allocate or use part of the 80GB buffer
            // For 1M particles, we only need ~24MB, so the 80GB buffer is overkill but fine.
            // Let's initialize positions/velocities in the buffer
            ParticleState* pParticles = static_cast<ParticleState*>(pContents);
            for (uint32_t i = 0; i < numParticles; ++i) {
                pParticles[i].pos[0] = (float)i / numParticles;
                pParticles[i].pos[1] = 0.0f;
                pParticles[i].pos[2] = 0.0f;
                pParticles[i].vel[0] = 0.0f;
                pParticles[i].vel[1] = 1.0f; // Uniform initial velocity
                pParticles[i].vel[2] = 0.0f;
            }

            float B_field[3] = {0.0f, 0.0f, 1.0f}; // Z-axis magnetic field
            float E_field[3] = {0.1f, 0.0f, 0.0f}; // X-axis electric field

            MTL::Buffer* pBBuffer = pDevice->newBuffer(sizeof(B_field), MTL::ResourceStorageModeShared);
            MTL::Buffer* pEBuffer = pDevice->newBuffer(sizeof(E_field), MTL::ResourceStorageModeShared);
            MTL::Buffer* pDtBuffer = pDevice->newBuffer(sizeof(dt), MTL::ResourceStorageModeShared);

            memcpy(pBBuffer->contents(), B_field, sizeof(B_field));
            memcpy(pEBuffer->contents(), E_field, sizeof(E_field));
            memcpy(pDtBuffer->contents(), &dt, sizeof(dt));

            MTL::CommandQueue* pQueue = pDevice->newCommandQueue();
            
            std::cout << "Starting simulation: " << numParticles << " particles, " << numSteps << " steps..." << std::endl;
            
            auto start = std::chrono::high_resolution_clock::now();

            for (uint32_t step = 0; step < numSteps; ++step) {
                NS::AutoreleasePool* pLoopPool = NS::AutoreleasePool::alloc()->init();
                
                MTL::CommandBuffer* pCmdBuf = pQueue->commandBuffer();
                MTL::ComputeCommandEncoder* pEncoder = pCmdBuf->computeCommandEncoder();
                
                pEncoder->setComputePipelineState(pPSO);
                pEncoder->setBuffer(pStateBuffer, 0, 0);
                pEncoder->setBuffer(pBBuffer, 0, 1);
                pEncoder->setBuffer(pEBuffer, 0, 2);
                pEncoder->setBuffer(pDtBuffer, 0, 3);
                
                MTL::Size gridSize = MTL::Size::Make(numParticles, 1, 1);
                NS::UInteger threadgroupSizeVal = pPSO->maxTotalThreadsPerThreadgroup();
                if (threadgroupSizeVal > numParticles) threadgroupSizeVal = numParticles;
                MTL::Size threadgroupSize = MTL::Size::Make(threadgroupSizeVal, 1, 1);
                
                pEncoder->dispatchThreads(gridSize, threadgroupSize);
                pEncoder->endEncoding();
                pCmdBuf->commit();
                
                // For performance, we don't wait every step. 
                // But for the hero run, we want stability. 
                // Let's synchronize every 1000 steps to report progress.
                if (step % 10000 == 0 && step > 0) {
                    pCmdBuf->waitUntilCompleted();
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = now - start;
                    double rate = (double)numParticles * 10000 / diff.count();
                    std::cout << "[Step " << step << "/" << numSteps << "] | Rate: " << std::fixed << std::setprecision(0) << rate << " pt-up/s" << std::endl;
                    start = now;
                }
                
                pLoopPool->release();
            }

            std::cout << "Hero Loop COMPLETE." << std::endl;
            
            pQueue->release();
            pBBuffer->release();
            pEBuffer->release();
            pDtBuffer->release();
            pPSO->release();
        } else {
            std::cerr << "Error: Failed to create Compute Pipeline State." << std::endl;
            if (pError) std::cerr << "Metal Error: " << pError->localizedDescription()->utf8String() << std::endl;
        }
        pFunc->release();
        pLibrary->release();
    }
    pOptions->release();

    pStateBuffer->release();
    pDevice->release();
    pPool->release();
    
    return 0;
}
