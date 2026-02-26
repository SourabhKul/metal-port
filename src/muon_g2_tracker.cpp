#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

struct MuonState {
    float pos[4];
    float vel[4];
    float spin[4];
};

int main(int argc, char* argv[]) {
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();

    uint32_t numMuons = 1000000; // 1M muons for initial test
    if (argc > 1) numMuons = std::stoi(argv[1]);
    
    const uint32_t numSteps = 10000;
    const float dt = 0.75e-9f; // 0.75 ns per step (g-2 spec)

    std::cout << "--- Metal C++: Muon g-2 Coupled Spin-Orbit Tracker ---" << std::endl;
    std::cout << "Muons: " << numMuons << " | Steps: " << numSteps << " | dt: " << dt << "s" << std::endl;

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

    MTL::Function* pFunc = pLibrary->newFunction(NS::String::string("muon_g2_kernel", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pPSO = pDevice->newComputePipelineState(pFunc, &pError);

    // Buffers
    MTL::Buffer* pStateBuffer = pDevice->newBuffer(numMuons * sizeof(MuonState), MTL::ResourceStorageModeShared);
    float B_field[4] = {0.0f, 0.0f, 1.45f, 0.0f}; // 1.45T storage ring field
    float E_field[4] = {0.0f, 0.0f, 0.0f, 0.0f}; 
    MTL::Buffer* pBBuffer = pDevice->newBuffer(sizeof(float)*4, MTL::ResourceStorageModeShared);
    MTL::Buffer* pEBuffer = pDevice->newBuffer(sizeof(float)*4, MTL::ResourceStorageModeShared);
    MTL::Buffer* pDtBuffer = pDevice->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    memcpy(pBBuffer->contents(), B_field, sizeof(float)*4);
    memcpy(pEBuffer->contents(), E_field, sizeof(float)*4);
    memcpy(pDtBuffer->contents(), &dt, sizeof(float));

    // Init: muons at rest or relativistic?
    // Magic momentum p = 3.094 GeV/c, gamma = 29.3
    float v_magic = 299792458.0f * 0.9994f; // approx beta=0.9994
    MuonState* pMuons = static_cast<MuonState*>(pStateBuffer->contents());
    for(uint32_t i=0; i<numMuons; ++i) {
        pMuons[i].pos[0] = 7112.0f; // 7.112m radius
        pMuons[i].pos[1] = 0.0f; pMuons[i].pos[2] = 0.0f; pMuons[i].pos[3] = 0.0f;
        pMuons[i].vel[0] = 0.0f; pMuons[i].vel[1] = v_magic; pMuons[i].vel[2] = 0.0f; pMuons[i].vel[3] = 0.0f;
        pMuons[i].spin[0] = 0.0f; pMuons[i].spin[1] = 1.0f; pMuons[i].spin[2] = 0.0f; pMuons[i].spin[3] = 0.0f;
    }

    MTL::CommandQueue* pQueue = pDevice->newCommandQueue();
    auto start = std::chrono::high_resolution_clock::now();

    const uint32_t stepsPerBatch = 100;
    const uint32_t numBatches = numSteps / stepsPerBatch;

    for (uint32_t b = 0; b < numBatches; ++b) {
        NS::AutoreleasePool* pBatchPool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* pCmdBuf = pQueue->commandBuffer();
        
        for(uint32_t step = 0; step < stepsPerBatch; ++step) {
            MTL::ComputeCommandEncoder* pEncoder = pCmdBuf->computeCommandEncoder();
            pEncoder->setComputePipelineState(pPSO);
            pEncoder->setBuffer(pStateBuffer, 0, 0);
            pEncoder->setBuffer(pBBuffer, 0, 1);
            pEncoder->setBuffer(pEBuffer, 0, 2);
            pEncoder->setBuffer(pDtBuffer, 0, 3);
            pEncoder->dispatchThreads(MTL::Size::Make(numMuons, 1, 1), MTL::Size::Make(256, 1, 1));
            pEncoder->endEncoding();
        }
        
        pCmdBuf->commit();
        pCmdBuf->waitUntilCompleted();
        pBatchPool->release();
        
        if (b % 10 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> d = now - start;
            double rate = (double)numMuons * (b + 1) * stepsPerBatch / d.count();
            std::cout << "[Batch " << b << "/" << numBatches << "] Rate: " << rate / 1e9 << " B muon-up/s" << std::endl;
        }
    }

    pQueue->commandBuffer()->commit();
    pQueue->commandBuffer()->waitUntilCompleted();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double rate = (double)numMuons * numSteps / diff.count();
    
    std::cout << "Muon Tracking Complete in " << diff.count() << "s" << std::endl;
    std::cout << "Throughput: " << rate / 1e9 << " Billion muon-updates/s" << std::endl;

    return 0;
}
