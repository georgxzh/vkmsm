// Reusable GPU Pippenger MSM engine (Stage 6), factored out so both the
// correctness test (stage6_pippenger_shader_test.cpp) and later stages
// that need to run it repeatedly (the Stage 7 benchmark harness, and
// Stage 8's window-size tuning) don't each duplicate the Vulkan setup.

#pragma once

#include <cstdint>
#include <vector>

#include "msm.hpp"
#include "point.hpp"

namespace vkmsm {

struct GpuPippengerContext;

// Opaque handle owning the Vulkan instance/device/pipeline/buffers needed
// to run GPU Pippenger. window_bits is fixed for the lifetime of a
// context (it determines the bucket buffer size); create a new context to
// benchmark a different window_bits. Plain create/destroy (not
// std::unique_ptr) because GpuPippengerContext is intentionally kept
// incomplete in this header - a unique_ptr's default deleter needs a
// complete type wherever it's destroyed, which would leak the Vulkan
// internals into every translation unit that just wants to call this API.
GpuPippengerContext* create_gpu_pippenger_context(int window_bits);
void destroy_gpu_pippenger_context(GpuPippengerContext* ctx);

// Also reports the device name the context is running on, for the
// benchmark report (Stage 7).
const char* gpu_pippenger_device_name(const GpuPippengerContext& ctx);

// Runs GPU Pippenger MSM for one (points, scalars) input, using the
// window_bits the context was created with. Combines per-window bucket
// sums on the CPU via the already-validated Stage 2 point_add.
PointJacobian gpu_pippenger(GpuPippengerContext& ctx, const std::vector<PointJacobian>& points,
                             const std::vector<Scalar>& scalars);

}  // namespace vkmsm
