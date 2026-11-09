#include "models/musc.hpp"
#include <array>
#include <cmath>

namespace musc {
static Metrics ranking(Tensor truth,Tensor scores) {
    truth=truth.flatten().contiguous().to(at::kByte);
    scores=scores.flatten().contiguous().to(at::kFloat);
    const int64_t pos=truth.sum().item<int64_t>(), neg=truth.numel()-pos;
    Metrics m;
    if(!pos || !neg) return m;
    auto [values,indices]=scores.sort(0,true);
    auto labels=truth.index_select(0,indices);
    auto y=labels.data_ptr<uint8_t>(); auto x=values.data_ptr<float>();
    int64_t tp=0,fp=0;
    for(int64_t i=0;i<scores.numel();) {
        auto prev_tp=tp,prev_fp=fp;
        auto score=x[i];
        do {if(y[i]) ++tp; else ++fp; ++i;} while(i<scores.numel() && x[i]==score);
        m.auc+=double(fp-prev_fp)*(tp+prev_tp)/(2.0*pos*neg);
        m.ap+=double(tp-prev_tp)/pos*tp/(tp+fp);
        m.f1=std::max(m.f1,2.0*tp/(pos+tp+fp));
    }
    return m;
}
static double pro(const Tensor& masks,const Tensor& scores) {
    const double lo=scores.min().item<double>(),hi=scores.max().item<double>();
    if(!(hi>lo)) return 0;
    const int side=static_cast<int>(masks.size(1)),area=side*side;
    std::array<double,201> overlap{},fp{};
    int64_t regions=0,background=0;
    auto mask=masks.data_ptr<uint8_t>(); auto val=scores.data_ptr<float>();
    auto bin=[&](double v){return std::clamp(static_cast<int>(std::ceil((v-lo)*200/(hi-lo))),0,200);};
    for(int64_t image=0;image<masks.size(0);++image,mask+=area,val+=area) {
        std::vector<uint8_t> seen(area);
        std::vector<int> region;
        for(int idx=0;idx<area;++idx) {
            if(!mask[idx]) {++background; ++fp[bin(val[idx])]; continue;}
            if(seen[idx]) continue;
            region.clear(); region.push_back(idx); seen[idx]=1;
            for(size_t j=0;j<region.size();++j) {
                int y=region[j]/side,x=region[j]%side;
                for(int ny=std::max(y-1,0);ny<=std::min(y+1,side-1);++ny)
                    for(int nx=std::max(x-1,0);nx<=std::min(x+1,side-1);++nx) {
                        int q=ny*side+nx;
                        if(mask[q] && !seen[q]) {seen[q]=1;region.push_back(q);}
                    }
            }
            ++regions;
            double weight=1.0/region.size();
            for(int q:region) overlap[bin(val[q])]+=weight;
        }
    }
    if(!regions || !background) return 0;
    double found=0,false_pos=0,integral=0,x0=-1,prev_x=0,prev_y=0;
    for(int j=200;j>=1;--j) {
        found+=overlap[j]; false_pos+=fp[j];
        auto x=false_pos/background,y=found/regions;
        if(x>=0.3) break;
        if(x0<0) x0=x;
        else integral+=(x-prev_x)*(prev_y+y)*0.5;
        prev_x=x;prev_y=y;
    }
    return prev_x>x0 && x0>=0?integral/(prev_x-x0):0;
}
Result evaluate(const std::vector<Sample>& samples,const Tensor& maps,const Tensor& scores,bool visa) {
    std::vector<uint8_t> labels;
    std::vector<Tensor> gt;
    for(auto& s:samples) {
        labels.push_back(s.abnormal);
        gt.push_back(read_mask(s,static_cast<int>(maps.size(1)),visa));
    }
    auto masks=at::stack(gt);
    return {ranking(at::tensor(labels),scores),ranking(masks,maps),pro(masks,maps)};
}
}
