// Stage 0 sanity check: prove the full Vulkan compute pipeline works
// (instance -> device -> compute queue -> shader -> descriptor sets ->
// command buffer -> submit -> sync -> readback) before any cryptography
// code is written. Runs c[i] = a[i] + b[i] on the GPU and checks it
// against the CPU-expected result.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                        \
    do {                                                                   \
        VkResult err_ = (x);                                               \
        if (err_ != VK_SUCCESS) {                                          \
            throw std::runtime_error(std::string("Vulkan error ") +        \
                                      std::to_string(err_) + " at " +       \
                                      __FILE__ + ":" +                     \
                                      std::to_string(__LINE__));            \
        }                                                                  \
    } while (0)

static const uint32_t N = 1024;
static const uint32_t WORKGROUP_SIZE = 256;

static std::vector<uint32_t> readSpirv(const char* path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open shader file: ") + path);
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    file.close();

    std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
    std::memcpy(spirv.data(), buffer.data(), fileSize);
    return spirv;
}

static uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
                                VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable GPU memory type");
}

static void createBuffer(VkDevice device, VkPhysicalDevice physDevice, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                          VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(physDevice, memRequirements.memoryTypeBits, properties);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &memory));

    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));
}

int main() {
    try {
        // --- 1. Instance: the entry point into the Vulkan API ---
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "vkmsm-stage0";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "none";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;

        VkInstance instance;
        VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));

        // --- 2. Pick a physical device (GPU) with a compute-capable queue family ---
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) throw std::runtime_error("No Vulkan-capable GPU found");
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        uint32_t computeQueueFamily = UINT32_MAX;
        for (auto& dev : devices) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());
            for (uint32_t i = 0; i < queueFamilyCount; i++) {
                if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = dev;
                    computeQueueFamily = i;
                    break;
                }
            }
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        if (physicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("No compute-capable queue family found on any GPU");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "Using GPU: " << props.deviceName << " (queue family "
                  << computeQueueFamily << ")\n";

        // --- 3. Logical device + compute queue handle ---
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = computeQueueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;

        VkDevice device;
        VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device));

        VkQueue computeQueue;
        vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

        // --- 4. Buffers: 3 GPU-visible float arrays (a, b, c) ---
        // HOST_VISIBLE | HOST_COHERENT lets the CPU read/write them directly
        // without a separate staging buffer. Fine for a sanity check; a real
        // discrete GPU workload would prefer device-local memory + staging.
        VkDeviceSize bufferSize = N * sizeof(float);
        VkBuffer bufA, bufB, bufC;
        VkDeviceMemory memA, memB, memC;
        VkMemoryPropertyFlags hostProps =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     hostProps, bufA, memA);
        createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     hostProps, bufB, memB);
        createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     hostProps, bufC, memC);

        // Fill input data: a[i] = i, b[i] = 2i
        {
            void* data;
            VK_CHECK(vkMapMemory(device, memA, 0, bufferSize, 0, &data));
            float* fa = reinterpret_cast<float*>(data);
            for (uint32_t i = 0; i < N; i++) fa[i] = static_cast<float>(i);
            vkUnmapMemory(device, memA);

            VK_CHECK(vkMapMemory(device, memB, 0, bufferSize, 0, &data));
            float* fb = reinterpret_cast<float*>(data);
            for (uint32_t i = 0; i < N; i++) fb[i] = static_cast<float>(i) * 2.0f;
            vkUnmapMemory(device, memB);
        }

        // --- 5. Load the compiled shader (SPIR-V bytecode) ---
        auto spirv = readSpirv(SHADER_PATH);
        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
        shaderInfo.pCode = spirv.data();
        VkShaderModule shaderModule;
        VK_CHECK(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule));

        // --- 6. Descriptor set layout: describes the 3 buffer bindings the shader expects ---
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int i = 0; i < 3; i++) {
            bindings[i].binding = static_cast<uint32_t>(i);
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        VkDescriptorSetLayout descSetLayout;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descSetLayout));

        // --- 7. Descriptor pool + set: actually points the bindings at our buffers ---
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        VkDescriptorPool descPool;
        VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool));

        VkDescriptorSetAllocateInfo dsAllocInfo{};
        dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAllocInfo.descriptorPool = descPool;
        dsAllocInfo.descriptorSetCount = 1;
        dsAllocInfo.pSetLayouts = &descSetLayout;
        VkDescriptorSet descSet;
        VK_CHECK(vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet));

        VkDescriptorBufferInfo bufInfos[3] = {
            {bufA, 0, VK_WHOLE_SIZE}, {bufB, 0, VK_WHOLE_SIZE}, {bufC, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; i++) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descSet;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufInfos[i];
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        // --- 8. Compute pipeline: shader + layout, ready to be dispatched ---
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descSetLayout;
        VkPipelineLayout pipelineLayout;
        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        VkPipeline pipeline;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                           &pipeline));

        // --- 9. Command pool + buffer: record the work to submit to the GPU ---
        VkCommandPoolCreateInfo cmdPoolInfo{};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = computeQueueFamily;
        VkCommandPool cmdPool;
        VK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = cmdPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        VkCommandBuffer cmdBuf;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuf));

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));

        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                 &descSet, 0, nullptr);
        vkCmdDispatch(cmdBuf, N / WORKGROUP_SIZE, 1, 1);

        VK_CHECK(vkEndCommandBuffer(cmdBuf));

        // --- 10. Submit to the GPU and wait for completion (synchronization) ---
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;
        VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        // --- 11. Read back results and verify against CPU-expected values ---
        void* data;
        VK_CHECK(vkMapMemory(device, memC, 0, bufferSize, 0, &data));
        float* fc = reinterpret_cast<float*>(data);
        bool ok = true;
        uint32_t mismatchIndex = 0;
        float gotVal = 0, expVal = 0;
        for (uint32_t i = 0; i < N; i++) {
            float expected = static_cast<float>(i) + static_cast<float>(i) * 2.0f;
            if (fc[i] != expected) {
                ok = false;
                mismatchIndex = i;
                gotVal = fc[i];
                expVal = expected;
                break;
            }
        }
        float c0 = fc[0], c1 = fc[1], cLast = fc[N - 1];
        vkUnmapMemory(device, memC);

        std::cout << "N = " << N << "\n";
        std::cout << "Sample: c[0]=" << c0 << " c[1]=" << c1 << " c[" << N - 1 << "]=" << cLast
                  << "\n";
        if (!ok) {
            std::cout << "Mismatch at index " << mismatchIndex << ": got " << gotVal
                      << ", expected " << expVal << "\n";
        }
        std::cout << (ok ? "PASS" : "FAIL") << ": GPU vector-add "
                  << (ok ? "matches" : "does NOT match") << " CPU expectation for " << N
                  << " elements\n";

        // --- Cleanup: destroy everything in reverse order of creation ---
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(device, descPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        vkDestroyBuffer(device, bufA, nullptr);
        vkFreeMemory(device, memA, nullptr);
        vkDestroyBuffer(device, bufB, nullptr);
        vkFreeMemory(device, memB, nullptr);
        vkDestroyBuffer(device, bufC, nullptr);
        vkFreeMemory(device, memC, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);

        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
