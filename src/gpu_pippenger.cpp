#include "gpu_pippenger.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "fp.hpp"

namespace vkmsm {
namespace {

constexpr uint32_t kWorkgroupSize = 64;
constexpr uint32_t kPointLimbs = kLimbs * 3;
constexpr uint32_t kScalarLimbs = 8;
constexpr int kScalarBits = 256;

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err_ = (x);                                            \
        if (err_ != VK_SUCCESS) {                                       \
            throw std::runtime_error(std::string("Vulkan error ") +     \
                                      std::to_string(err_) + " at " +    \
                                      __FILE__ + ":" +                  \
                                      std::to_string(__LINE__));         \
        }                                                                \
    } while (0)

std::vector<uint32_t> readSpirv(const char* path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error(std::string("Failed to open shader file: ") + path);
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
    std::memcpy(spirv.data(), buffer.data(), fileSize);
    return spirv;
}

uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter,
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

void createBuffer(VkDevice device, VkPhysicalDevice physDevice, VkDeviceSize size,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
                   VkDeviceMemory& memory) {
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
    allocInfo.memoryTypeIndex = findMemoryType(physDevice, memRequirements.memoryTypeBits, properties);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));
}

void pack_point(uint32_t* dst, const PointJacobian& p) {
    std::memcpy(dst, p.x.limb.data(), kLimbs * sizeof(uint32_t));
    std::memcpy(dst + kLimbs, p.y.limb.data(), kLimbs * sizeof(uint32_t));
    std::memcpy(dst + 2 * kLimbs, p.z.limb.data(), kLimbs * sizeof(uint32_t));
}

PointJacobian unpack_point(const uint32_t* src) {
    PointJacobian p;
    std::memcpy(p.x.limb.data(), src, kLimbs * sizeof(uint32_t));
    std::memcpy(p.y.limb.data(), src + kLimbs, kLimbs * sizeof(uint32_t));
    std::memcpy(p.z.limb.data(), src + 2 * kLimbs, kLimbs * sizeof(uint32_t));
    return p;
}

struct PushConstants {
    uint32_t n0;
    uint32_t modulus[kLimbs];
    uint32_t window_index;
    uint32_t window_bits;
    uint32_t num_points;
    uint32_t num_buckets;
};

}  // namespace

struct GpuPippengerContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamily;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descSetLayout;
    VkDescriptorPool descPool;
    VkDescriptorSet descSet;
    VkCommandPool cmdPool;
    VkCommandBuffer cmdBuf;
    VkFence fence;
    VkShaderModule shaderModule;

    VkBuffer bucketsBuf;
    VkDeviceMemory bucketsMem;
    uint32_t numBuckets;
    int windowBits;

    std::string deviceName;
};

GpuPippengerContext* create_gpu_pippenger_context(int window_bits) {
    auto* ctx = new GpuPippengerContext{};
    ctx->windowBits = window_bits;
    ctx->numBuckets = 1u << window_bits;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vkmsm-gpu-pippenger";
    appInfo.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &ctx->instance));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, devices.data());

    ctx->physicalDevice = VK_NULL_HANDLE;
    ctx->computeQueueFamily = UINT32_MAX;
    for (auto& dev : devices) {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qFamilies(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, qFamilies.data());
        for (uint32_t i = 0; i < qCount; i++) {
            if (qFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx->physicalDevice = dev;
                ctx->computeQueueFamily = i;
                break;
            }
        }
        if (ctx->physicalDevice != VK_NULL_HANDLE) break;
    }
    if (ctx->physicalDevice == VK_NULL_HANDLE) throw std::runtime_error("No compute-capable GPU found");

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physicalDevice, &props);
    ctx->deviceName = props.deviceName;

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = ctx->computeQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    VK_CHECK(vkCreateDevice(ctx->physicalDevice, &deviceInfo, nullptr, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->computeQueueFamily, 0, &ctx->computeQueue);

    VkDeviceSize bucketsSize = static_cast<VkDeviceSize>(ctx->numBuckets) * kPointLimbs * sizeof(uint32_t);
    VkMemoryPropertyFlags hostProps =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    createBuffer(ctx->device, ctx->physicalDevice, bucketsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 hostProps, ctx->bucketsBuf, ctx->bucketsMem);

    auto spirv = readSpirv(SHADER_PATH_PIPPENGER_BUCKETS);
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
    shaderInfo.pCode = spirv.data();
    VK_CHECK(vkCreateShaderModule(ctx->device, &shaderInfo, nullptr, &ctx->shaderModule));

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
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, nullptr, &ctx->descSetLayout));

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    VK_CHECK(vkCreateDescriptorPool(ctx->device, &poolInfo, nullptr, &ctx->descPool));

    VkDescriptorSetAllocateInfo dsAllocInfo{};
    dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAllocInfo.descriptorPool = ctx->descPool;
    dsAllocInfo.descriptorSetCount = 1;
    dsAllocInfo.pSetLayouts = &ctx->descSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(ctx->device, &dsAllocInfo, &ctx->descSet));

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &ctx->descSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(ctx->device, &pipelineLayoutInfo, nullptr, &ctx->pipelineLayout));

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = ctx->shaderModule;
    stageInfo.pName = "main";
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = ctx->pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &ctx->pipeline));

    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = ctx->computeQueueFamily;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(ctx->device, &cmdPoolInfo, nullptr, &ctx->cmdPool));

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = ctx->cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &ctx->cmdBuf));

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(ctx->device, &fenceInfo, nullptr, &ctx->fence));

    return ctx;
}

void destroy_gpu_pippenger_context(GpuPippengerContext* ctx) {
    if (!ctx) return;
    vkDestroyFence(ctx->device, ctx->fence, nullptr);
    vkDestroyCommandPool(ctx->device, ctx->cmdPool, nullptr);
    vkDestroyPipeline(ctx->device, ctx->pipeline, nullptr);
    vkDestroyPipelineLayout(ctx->device, ctx->pipelineLayout, nullptr);
    vkDestroyDescriptorPool(ctx->device, ctx->descPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx->device, ctx->descSetLayout, nullptr);
    vkDestroyShaderModule(ctx->device, ctx->shaderModule, nullptr);
    vkDestroyBuffer(ctx->device, ctx->bucketsBuf, nullptr);
    vkFreeMemory(ctx->device, ctx->bucketsMem, nullptr);
    vkDestroyDevice(ctx->device, nullptr);
    vkDestroyInstance(ctx->instance, nullptr);
    delete ctx;
}

const char* gpu_pippenger_device_name(const GpuPippengerContext& ctx) { return ctx.deviceName.c_str(); }

PointJacobian gpu_pippenger(GpuPippengerContext& ctx, const std::vector<PointJacobian>& points,
                             const std::vector<Scalar>& scalars) {
    uint32_t n = static_cast<uint32_t>(points.size());
    if (n == 0) return point_infinity();

    VkDeviceSize pointsSize = static_cast<VkDeviceSize>(n) * kPointLimbs * sizeof(uint32_t);
    VkDeviceSize scalarsSize = static_cast<VkDeviceSize>(n) * kScalarLimbs * sizeof(uint32_t);
    VkMemoryPropertyFlags hostProps =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBuffer pointsBuf, scalarsBuf;
    VkDeviceMemory pointsMem, scalarsMem;
    createBuffer(ctx.device, ctx.physicalDevice, pointsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 hostProps, pointsBuf, pointsMem);
    createBuffer(ctx.device, ctx.physicalDevice, scalarsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 hostProps, scalarsBuf, scalarsMem);

    void* data;
    VK_CHECK(vkMapMemory(ctx.device, pointsMem, 0, pointsSize, 0, &data));
    auto* pp = reinterpret_cast<uint32_t*>(data);
    for (uint32_t i = 0; i < n; i++) pack_point(pp + i * kPointLimbs, points[i]);
    vkUnmapMemory(ctx.device, pointsMem);

    VK_CHECK(vkMapMemory(ctx.device, scalarsMem, 0, scalarsSize, 0, &data));
    auto* ps = reinterpret_cast<uint32_t*>(data);
    for (uint32_t i = 0; i < n; i++) std::memcpy(ps + i * kScalarLimbs, scalars[i].data(), kScalarLimbs * sizeof(uint32_t));
    vkUnmapMemory(ctx.device, scalarsMem);

    VkDescriptorBufferInfo bufInfos[3] = {
        {pointsBuf, 0, VK_WHOLE_SIZE}, {scalarsBuf, 0, VK_WHOLE_SIZE}, {ctx.bucketsBuf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ctx.descSet;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

    PushConstants pc{};
    pc.n0 = fp_n0();
    for (int i = 0; i < kLimbs; i++) pc.modulus[i] = fp_modulus().limb[i];
    pc.window_bits = static_cast<uint32_t>(ctx.windowBits);
    pc.num_points = n;
    pc.num_buckets = ctx.numBuckets;

    int num_windows = (kScalarBits + ctx.windowBits - 1) / ctx.windowBits;
    VkDeviceSize bucketsSize = static_cast<VkDeviceSize>(ctx.numBuckets) * kPointLimbs * sizeof(uint32_t);

    PointJacobian result = point_infinity();

    for (int w = num_windows - 1; w >= 0; w--) {
        if (w != num_windows - 1) {
            for (int i = 0; i < ctx.windowBits; i++) result = point_double(result);
        }
        pc.window_index = static_cast<uint32_t>(w);

        VK_CHECK(vkResetCommandBuffer(ctx.cmdBuf, 0));
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(ctx.cmdBuf, &beginInfo));
        vkCmdBindPipeline(ctx.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
        vkCmdBindDescriptorSets(ctx.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipelineLayout, 0,
                                 1, &ctx.descSet, 0, nullptr);
        vkCmdPushConstants(ctx.cmdBuf, ctx.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(PushConstants), &pc);
        uint32_t groups = (ctx.numBuckets + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(ctx.cmdBuf, groups, 1, 1);
        VK_CHECK(vkEndCommandBuffer(ctx.cmdBuf));

        VK_CHECK(vkResetFences(ctx.device, 1, &ctx.fence));
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &ctx.cmdBuf;
        VK_CHECK(vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, ctx.fence));
        VK_CHECK(vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX));

        VK_CHECK(vkMapMemory(ctx.device, ctx.bucketsMem, 0, bucketsSize, 0, &data));
        auto* pb = reinterpret_cast<uint32_t*>(data);
        std::vector<PointJacobian> buckets(ctx.numBuckets);
        for (uint32_t k = 0; k < ctx.numBuckets; k++) buckets[k] = unpack_point(pb + k * kPointLimbs);
        vkUnmapMemory(ctx.device, ctx.bucketsMem);

        PointJacobian running_sum = point_infinity();
        PointJacobian window_total = point_infinity();
        for (uint32_t k = ctx.numBuckets - 1; k >= 1; k--) {
            running_sum = point_add(running_sum, buckets[k]);
            window_total = point_add(window_total, running_sum);
        }
        result = point_add(result, window_total);
    }

    vkDestroyBuffer(ctx.device, pointsBuf, nullptr);
    vkFreeMemory(ctx.device, pointsMem, nullptr);
    vkDestroyBuffer(ctx.device, scalarsBuf, nullptr);
    vkFreeMemory(ctx.device, scalarsMem, nullptr);

    return result;
}

}  // namespace vkmsm
