#include "models/musc.hpp"
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <windows.h>
#include <cmath>
#include <limits>
#include "musc_match_ptx.hpp"

namespace musc {
namespace {
class Kernel {
    using Load = int(WINAPI*)(void**, const void*);
    using Find = int(WINAPI*)(void**, void*, const char*);
    using Unload = int(WINAPI*)(void*);
    using Launch = int(WINAPI*)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
    void* module = nullptr;
    Unload unload = nullptr;
    Launch launch = nullptr;
public:
    void* function = nullptr;
    Kernel() {
        auto driver = GetModuleHandleW(L"nvcuda.dll");
        if (!driver) return;
        auto load = reinterpret_cast<Load>(GetProcAddress(driver, "cuModuleLoadData"));
        auto find = reinterpret_cast<Find>(GetProcAddress(driver, "cuModuleGetFunction"));
        unload = reinterpret_cast<Unload>(GetProcAddress(driver, "cuModuleUnload"));
        launch = reinterpret_cast<Launch>(GetProcAddress(driver, "cuLaunchKernel"));
        if (!load || !find || !unload || !launch) return;
        if (load(&module, tiled_ptx) != 0 || find(&function, module, "tiled_min") != 0)
            function = nullptr;
    }
    ~Kernel() {
        if (module) unload(module);
    }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    void run(unsigned tiles, unsigned references, void** args) const {
        int status = launch(function, tiles, tiles, references, 16, 16, 1, 0, nullptr, args, nullptr);
        TORCH_CHECK_WITH(OutOfMemoryError, status != 2, "CUDA tiled matching ran out of memory");
        TORCH_CHECK(status == 0, "CUDA tiled matching launch failed: ", status);
    }
};
}

Tensor tiled_scores(const Tensor& features, int64_t k) {
    if (features.scalar_type() != at::kFloat || features.size(2) % 16 != 0) return {};
    if (c10::impl::VirtualGuardImpl(at::kCUDA).getStream(features.device()).id() != 0) return {};
    const double images_count = features.size(0), patches_count = features.size(1);
    const double channels_count = features.size(2), stride_count = std::ceil(patches_count / 128) * 128;
    const double packed_bytes = images_count * channels_count * stride_count * sizeof(float);
    const double minima_bytes = images_count * images_count * patches_count * sizeof(float);
    const double norms_bytes = images_count * stride_count * sizeof(float);
    const double topk_bytes = images_count * k * patches_count * (sizeof(float) + sizeof(int64_t));
    const double workspace = std::max(static_cast<double>(features.nbytes()), minima_bytes + topk_bytes);
    if (features.nbytes() + packed_bytes + norms_bytes + workspace > 2.0 * 1024 * 1024 * 1024) return {};
    c10::DeviceGuard device_guard(features.device());
    Kernel kernel;
    if (!kernel.function) {
        TORCH_WARN_ONCE("Fused CUDA matching is unavailable; using LibTorch matching.");
        return {};
    }
    int images = static_cast<int>(images_count), patches = static_cast<int>(patches_count);
    int channels = static_cast<int>(channels_count), stride = static_cast<int>(stride_count);
    auto packed = at::zeros({images, channels, stride}, features.options());
    packed.slice(2, 0, patches).copy_(features.transpose(1, 2));
    auto norms = at::full({images, stride}, std::numeric_limits<float>::infinity(), features.options());
    norms.slice(1, 0, patches).copy_(features.square().sum(-1));
    auto minima = at::full({images, images, patches}, std::numeric_limits<float>::infinity(), features.options());
    auto data = packed.data_ptr<float>();
    auto squared_norm = norms.data_ptr<float>();
    auto output = minima.data_ptr<float>();
    for (int i = 0; i + 1 < images; ++i) {
        for (int j = i + 1; j < images; j += 8) {
            void* args[] = {&data, &squared_norm, &output, &images, &patches, &stride, &channels, &i, &j};
            kernel.run(stride / 128, std::min(8, images - j), args);
        }
    }
    auto nearest = std::get<0>(minima.topk(k, 1, false, false));
    return nearest.sqrt_().mean(1).cpu();
}
}
