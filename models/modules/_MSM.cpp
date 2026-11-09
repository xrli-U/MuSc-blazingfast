#include "models/musc.hpp"
#include <limits>

namespace musc {
#ifdef MUSC_TILED_CUDA
Tensor tiled_scores(const Tensor&, int64_t);
#endif
namespace {
constexpr int64_t reference_batch = 8;

bool fits_resident_budget(const Tensor& z, int64_t k) {
    const double images = z.size(0), patches = z.size(1), channels = z.size(2);
    const double value_bytes = z.element_size();
    const double feature_bytes = z.nbytes();
    const double minima_bytes = images * images * patches * value_bytes;
    const double topk_bytes = images * k * patches * (value_bytes + sizeof(int64_t));
    const double distance_workspace = reference_batch * patches * (patches + 4 * (channels + 2)) * value_bytes;
    constexpr double budget = 2.0 * 1024 * 1024 * 1024;
    return feature_bytes + minima_bytes + std::max(topk_bytes, distance_workspace) <= budget;
}

Tensor mutual_resident(const Tensor& z, at::Device device, int64_t k) {
    auto features = z.to(device);
#ifdef MUSC_TILED_CUDA
    auto fused = tiled_scores(features, k);
    if (fused.defined()) return fused;
#endif
    const auto images = features.size(0), patches = features.size(1);
    auto minima = at::full({images, images, patches},
                           std::numeric_limits<double>::infinity(), features.options());
    for (int64_t i = 0; i + 1 < images; ++i) {
        auto query = features.slice(0, i, i + 1);
        for (int64_t j = i + 1; j < images; j += reference_batch) {
            auto end = std::min(j + reference_batch, images);
            auto distances = at::cdist(query, features.slice(0, j, end));
            minima[i].slice(0, j, end).copy_(distances.amin(2));
            minima.slice(0, j, end).select(1, i).copy_(distances.amin(1));
        }
    }
    auto nearest = std::get<0>(minima.topk(k, 1, false, false));
    return nearest.mean(1).cpu();
}

Tensor mutual_streamed(const Tensor& z, at::Device device, int64_t k, int64_t batch_size) {
    const auto n=z.size(0), patches=z.size(1);
    std::vector<Tensor> results;
    if(patches <= 64000000/n/k) {
        std::vector<Tensor> best(n);
        auto update=[&](int64_t image,Tensor values) {
            values=values.cpu();
            if(best[image].defined()) values=at::cat({best[image],values},0);
            best[image]=values.size(0)>k?std::get<0>(values.topk(k,0,false,false)):values;
        };
        for(int64_t i=0;i<n;++i) {
            auto query=z[i].to(device).unsqueeze(0);
            for(int64_t j=i+1;j<n;j+=batch_size) {
                auto dist=at::cdist(query,z.slice(0,j,std::min(j+batch_size,n)).to(device));
                update(i,std::get<0>(dist.min(2)));
                auto reverse=std::get<0>(dist.min(1)).cpu();
                for(int64_t t=0;t<reverse.size(0);++t) update(j+t,reverse.slice(0,t,t+1));
            }
            results.push_back(best[i].mean(0));
            best[i]=Tensor{};
        }
        return at::stack(results);
    }
    for(int64_t i=0;i<n;++i) {
        auto query=z[i].to(device);
        auto best=at::empty({0,patches},query.options());
        for(int64_t first=0;first<n;first+=batch_size) {
            auto last=std::min(first+batch_size,n);
            auto refs=z.slice(0,first,last).to(device);
            auto dist=at::cdist(query.unsqueeze(0),refs);
            auto mins=std::get<0>(dist.min(2));
            if(i>=first && i<last)
                mins=at::cat({mins.slice(0,0,i-first),mins.slice(0,i-first+1)},0);
            best=at::cat({best,mins},0);
            if(best.size(0)>k) best=std::get<0>(best.topk(k,0,false,false));
        }
        results.push_back(best.mean(0).cpu());
    }
    return at::stack(results);
}
}

Tensor mutual(const Tensor& z, at::Device device) {
    const auto images = z.size(0);
    TORCH_CHECK(images >= 2, "Mutual scoring requires at least two images per subset");
    const auto k = std::max<int64_t>(1, (images - 1) * 3 / 10);
    if (device.is_cuda() && fits_resident_budget(z, k)) {
        try {
            return mutual_resident(z, device, k);
        } catch (const c10::OutOfMemoryError&) {
            if (z.is_cuda()) throw;
            TORCH_WARN_ONCE("GPU memory is limited; switching to streamed mutual scoring.");
        }
    }
    int64_t batch_size = reference_batch;
    for (;;) {
        try {
            return mutual_streamed(z, device, k, batch_size);
        } catch (const c10::OutOfMemoryError&) {
            if (!device.is_cuda() || batch_size == 1) throw;
            batch_size /= 2;
        }
    }
}
}
