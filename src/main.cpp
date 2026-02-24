#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <iomanip>

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
