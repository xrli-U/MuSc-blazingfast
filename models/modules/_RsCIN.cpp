#include "models/musc.hpp"
namespace musc {
Tensor rescore(const Tensor& scores,const Tensor& cls,const std::vector<int>& windows) {
    auto s=(scores-scores.min())/(scores.max()-scores.min()).clamp_min(1e-12);
    auto sim=at::matmul(cls,cls.t());
    auto total=at::zeros_like(s);
    for(auto window:windows) {
        auto [w,id]=sim.topk(std::min<int64_t>(window,sim.size(1)),1);
        auto neighbors=s.index_select(0,id.flatten()).reshape_as(w);
        auto sum=w.sum(1);
        total+=at::where(sum.abs()>1e-12,(w*neighbors).sum(1)/sum,s);
    }
    return total/windows.size();
}
}
