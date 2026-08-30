// Stage 5 validation: our GLSL port of G1 Jacobian point arithmetic
// (shaders/point_ops.comp), cross-checked against the already-validated
// CPU implementation (src/point.cpp, Stage 2) - per the project's build
// order, Stage 5 validates against the CPU reference from the stage
// before it, not against blst directly.
//
// blst is used only to source random *input* points (via scalar
// multiplication of its generator) - the add/double operations under
// test are entirely our own CPU and GPU implementations, compared
// against each other.

#include <blst.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "fp.hpp"
#include "point.hpp"

using namespace vkmsm;

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

namespace {

constexpr uint32_t kN = 512;  // points per op
constexpr uint32_t kWorkgroupSize = 64;
constexpr uint32_t kPointLimbs = kLimbs * 3;  // X, Y, Z packed together

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    g_checks++;
    if (!condition) {
        g_failures++;
        std::printf("  FAIL: %s\n", what);
    }
}

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

PointJacobian from_blst(const blst_p1& p) {
    PointJacobian r;
    std::array<uint32_t, kLimbs> x, y, z;
    blst_uint32_from_fp(x.data(), &p.x);
    blst_uint32_from_fp(y.data(), &p.y);
    blst_uint32_from_fp(z.data(), &p.z);
    r.x = fp_from_plain_limbs(x);
    r.y = fp_from_plain_limbs(y);
    r.z = fp_from_plain_limbs(z);
    return r;
}

blst_p1 random_blst_point(std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist;
    uint8_t scalar[32];
    for (auto& b : scalar) b = static_cast<uint8_t>(dist(rng));
    blst_p1 r;
    blst_p1_mult(&r, blst_p1_generator(), scalar, 256);
    return r;
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
    uint32_t op;
    uint32_t n0;
    uint32_t modulus[kLimbs];
};

}  // namespace

int main() {
    try {
        std::printf("Stage 5: BLS12-381 G1 point arithmetic on GPU vs CPU (Stage 2 reference)\n\n");

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "vkmsm-stage5";
        appInfo.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;
        VkInstance instance;
        VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) throw std::runtime_error("No Vulkan-capable GPU found");
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        uint32_t computeQueueFamily = UINT32_MAX;
        for (auto& dev : devices) {
            uint32_t qCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
            std::vector<VkQueueFamilyProperties> qFamilies(qCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, qFamilies.data());
            for (uint32_t i = 0; i < qCount; i++) {
                if (qFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = dev;
                    computeQueueFamily = i;
                    break;
                }
            }
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        if (physicalDevice == VK_NULL_HANDLE) throw std::runtime_error("No compute-capable GPU found");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::printf("Using GPU: %s\n\n", props.deviceName);

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

        VkDeviceSize bufferSize = static_cast<VkDeviceSize>(kN) * kPointLimbs * sizeof(uint32_t);
        VkBuffer bufA, bufB, bufC;
        VkDeviceMemory memA, memB, memC;
        VkMemoryPropertyFlags hostProps =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, bufA, memA);
        createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, bufB, memB);
        createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostProps, bufC, memC);

        auto spirv = readSpirv(SHADER_PATH);
        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
        shaderInfo.pCode = spirv.data();
        VkShaderModule shaderModule;
        VK_CHECK(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule));

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

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
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
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

        VkCommandPoolCreateInfo cmdPoolInfo{};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = computeQueueFamily;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool cmdPool;
        VK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = cmdPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        VkCommandBuffer cmdBuf;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuf));

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

        PushConstants pc{};
        pc.n0 = fp_n0();
        for (int i = 0; i < kLimbs; i++) pc.modulus[i] = fp_modulus().limb[i];

        std::mt19937 rng(0xB16B00B5);

        auto run_op = [&](uint32_t op, const char* name, bool need_b) {
            pc.op = op;
            std::vector<PointJacobian> a_pts(kN), b_pts(kN);

            void* data;
            VK_CHECK(vkMapMemory(device, memA, 0, bufferSize, 0, &data));
            auto* pa = reinterpret_cast<uint32_t*>(data);
            for (uint32_t i = 0; i < kN; i++) {
                a_pts[i] = from_blst(random_blst_point(rng));
                pack_point(pa + i * kPointLimbs, a_pts[i]);
            }
            vkUnmapMemory(device, memA);

            if (need_b) {
                VK_CHECK(vkMapMemory(device, memB, 0, bufferSize, 0, &data));
                auto* pb = reinterpret_cast<uint32_t*>(data);
                for (uint32_t i = 0; i < kN; i++) {
                    b_pts[i] = from_blst(random_blst_point(rng));
                    pack_point(pb + i * kPointLimbs, b_pts[i]);
                }
                vkUnmapMemory(device, memB);
            }

            VK_CHECK(vkResetCommandBuffer(cmdBuf, 0));
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                     &descSet, 0, nullptr);
            vkCmdPushConstants(cmdBuf, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(PushConstants), &pc);
            vkCmdDispatch(cmdBuf, kN / kWorkgroupSize, 1, 1);
            VK_CHECK(vkEndCommandBuffer(cmdBuf));

            VK_CHECK(vkResetFences(device, 1, &fence));
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuf;
            VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, fence));
            VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

            VK_CHECK(vkMapMemory(device, memC, 0, bufferSize, 0, &data));
            auto* pcRes = reinterpret_cast<uint32_t*>(data);
            int op_ok = 0;
            for (uint32_t i = 0; i < kN; i++) {
                PointJacobian expected = need_b ? point_add(a_pts[i], b_pts[i]) : point_double(a_pts[i]);
                PointJacobian gpu_result = unpack_point(pcRes + i * kPointLimbs);

                bool ok = expected.x.limb == gpu_result.x.limb && expected.y.limb == gpu_result.y.limb &&
                          expected.z.limb == gpu_result.z.limb;
                check(ok, "GPU point result matches CPU reference (Stage 2)");
                if (ok) op_ok++;
            }
            vkUnmapMemory(device, memC);
            std::printf("GPU %s (%u points): %d/%u matched CPU\n", name, kN, op_ok, kN);
        };

        run_op(0, "double", false);
        run_op(1, "add", true);

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

        std::printf("\n=====================================\n");
        std::printf("Total: %d checks, %d failures\n", g_checks, g_failures);
        std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
        return g_failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
