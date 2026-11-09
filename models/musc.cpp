#include "models/musc.hpp"
#include <ATen/Parallel.h>
#include <windows.h>
#include <objbase.h>
#include <chrono>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>

namespace musc {
static Tensor infer_subset(const Config& c,Vision& model,at::Device device,const std::vector<Sample>& images,std::vector<Tensor>& globals) {
    TORCH_CHECK(images.size()>=2,"Each subset requires at least two test images. Reduce divide_num.");
    std::vector<std::vector<Tensor>> layers(c.layers.size());
    const bool dino = c.backbone.rfind("dino", 0) == 0;
    auto load_batch = [&](size_t start) {
        const size_t end = std::min(start + c.batch, images.size());
        std::vector<Tensor> batch;
        batch.reserve(end - start);
        for (size_t j = start; j < end; ++j)
            batch.push_back(read_image(images[j].image, c.size, dino));
        return at::stack(batch);
    };
    auto load_on_worker = [&](size_t start) {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        TORCH_CHECK(SUCCEEDED(hr), "Cannot initialize Windows image decoding on the prefetch thread: ", hr);
        struct ComScope {
            ~ComScope() { CoUninitialize(); }
        } apartment;
        at::InferenceMode guard;
        return load_batch(start);
    };
    std::future<Tensor> pending;
    for (size_t start = 0; start < images.size(); start += c.batch) {
        auto input = pending.valid() ? pending.get() : load_batch(start);
        const size_t next = start + c.batch;
        if (device.is_cuda() && next < images.size())
            pending = std::async(std::launch::async, load_on_worker, next);
        auto features = model(input);
        globals.push_back(features.global);
        for(size_t l=0;l<layers.size();++l) layers[l].push_back(features.patches[l]);
    }
    Tensor scores;
    for(auto& layer:layers) {
        std::vector<Tensor> cached;
        const auto& sample = layer.front();
        double elements = static_cast<double>(images.size()) * sample.size(1) * sample.size(2);
        if (device.is_cuda() && elements * (sample.element_size() + sizeof(float)) <= 2.0 * 1024 * 1024 * 1024) {
            try {
                for (const auto& batch : layer) cached.push_back(batch.to(device));
            } catch (const c10::OutOfMemoryError&) {
                cached.clear();
            }
        }
        for(auto r:c.radii) {
            Tensor result;
            if (!cached.empty()) {
                try {
                    auto pooled = at::empty({static_cast<int64_t>(images.size()), sample.size(1), sample.size(2)},
                                            cached.front().options().dtype(at::kFloat));
                    int64_t position = 0;
                    for (const auto& batch : cached) {
                        auto end = position + batch.size(0);
                        pooled.slice(0, position, end).copy_(aggregate(batch, r));
                        position = end;
                    }
                    result = mutual(pooled, device);
                } catch (const c10::OutOfMemoryError&) {
                    cached.clear();
                    TORCH_WARN_ONCE("GPU memory is limited; staging aggregated features in CPU memory.");
                }
            }
            if (!result.defined()) {
                std::vector<Tensor> pooled;
                for(auto& batch:layer) pooled.push_back(aggregate(batch.to(device),r).cpu());
                result=mutual(at::cat(pooled),device);
            }
            if(!scores.defined()) scores=result;
            else scores+=result;
        }
        layer.clear();
    }
    scores/=static_cast<double>(c.radii.size()*layers.size());
    int64_t grid=static_cast<int64_t>(std::sqrt(scores.size(1)));
    return at::upsample_bilinear2d(scores.reshape({-1,1,grid,grid}),{c.size,c.size},true).squeeze(1).contiguous();
}
void run(const Config& c) {
    auto dataset=read_dataset(c);
    at::Device device(at::kCPU);
    if(c.device!="cpu") {
        int index=std::stoi(c.device);
        LoadLibraryW(L"torch_cuda.dll");
        if(at::getNumGPUs()>0) {
            TORCH_CHECK(index>=0 && index<at::getNumGPUs(),"Unavailable GPU ",index);
            device=at::Device(at::kCUDA,static_cast<c10::DeviceIndex>(index));
        } else std::cerr<<"CUDA unavailable; using CPU\n";
    }
    at::set_num_threads(std::min(8,at::get_num_threads()));
    std::cout<<"device="<<device<<" model="<<c.backbone<<" image_size="<<c.size<<std::endl;
    Vision model(c,device);
    auto folder=c.output/c.dataset/c.backbone/("imagesize"+std::to_string(c.size));
    fs::create_directories(folder);
    std::map<std::string,Result> all;
    Result mean{};
    for(auto& [name,images]:dataset) {
        auto start=std::chrono::steady_clock::now();
        std::vector<Sample> samples;
        std::vector<Tensor> maps,globals;
        for(auto& subset:split_data(images,c.divisions)) {
            maps.push_back(infer_subset(c,model,device,subset,globals));
            samples.insert(samples.end(),subset.begin(),subset.end());
        }
        auto amap=at::cat(maps);
        const std::vector<int> windows=c.dataset=="visa"?std::vector<int>{1,8,9}:std::vector<int>{1,2,3};
        auto scores=rescore(std::get<0>(amap.flatten(1).max(1)),at::cat(globals),windows);
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        TORCH_CHECK(at::isfinite(scores).all().item<bool>() && at::isfinite(amap).all().item<bool>(),"Non-finite anomaly scores");
        auto result=evaluate(samples,amap,scores,c.dataset=="visa");
        all[name]=result;
        auto dest=folder/fs::u8path(name);
        fs::create_directories(dest);
        std::ofstream csv(dest/"scores.csv");
        csv<<"image,anomaly_score\n"<<std::setprecision(9);
        double lo=amap.min().item<double>(),hi=amap.max().item<double>();
        for(size_t i=0;i<samples.size();++i) {
            csv<<std::quoted(samples[i].image.u8string(),'"','"')<<','<<scores[i].item<double>()<<'\n';
            if(c.vis && (c.vis_type!="single_norm" || samples[i].abnormal)) {
                auto map=amap[i];
                auto file=dest/fs::u8path(samples[i].type)/samples[i].image.filename();
                heatmap(file.replace_extension(".png"),map,c.vis_type=="single_norm"?map.min().item<double>():lo,c.vis_type=="single_norm"?map.max().item<double>():hi);
            }
        }
        TORCH_CHECK(csv,"Could not write scores");
        std::cout<<name<<" images="<<samples.size()<<" inference_s="<<sec<<std::endl;
        auto add=[&](Metrics& sum,const Metrics& m){sum.auc+=m.auc/dataset.size();sum.f1+=m.f1/dataset.size();sum.ap+=m.ap/dataset.size();};
        add(mean.image,result.image);add(mean.pixel,result.pixel);mean.pro+=result.pro/dataset.size();
    }
    all["mean"]=mean;
    for(const auto& [name,r]:all) std::cout<<name<<" image(AUROC,F1,AP) "<<r.image.auc*100<<' '<<r.image.f1*100<<' '<<r.image.ap*100
        <<" pixel(AUROC,F1,AP,PRO) "<<r.pixel.auc*100<<' '<<r.pixel.f1*100<<' '<<r.pixel.ap*100<<' '<<r.pro*100<<'\n';
    if(c.excel) workbook(folder/"results.xlsx",all);
}
}
