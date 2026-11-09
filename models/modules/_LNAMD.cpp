#include "models/musc.hpp"
#include <cmath>
namespace musc {
Tensor aggregate(const Tensor& tokens,int radius) {
    auto side=static_cast<int64_t>(std::sqrt(tokens.size(1)));
    TORCH_CHECK(side*side==tokens.size(1),"Non-square patch grid");
    auto spatial=tokens.transpose(1,2).reshape({tokens.size(0),tokens.size(2),side,side}).to(at::kFloat);
    spatial=at::layer_norm(spatial,{tokens.size(2),side,side});
    if(radius>1) spatial=at::avg_pool2d(spatial,{radius,radius},{1,1},{radius/2,radius/2},false,true);
    auto features=spatial.flatten(2).transpose(1,2);
    return (features/features.norm(2,-1,true).clamp_min(1e-12)).contiguous();
}
}
