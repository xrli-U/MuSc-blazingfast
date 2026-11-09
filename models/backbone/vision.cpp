#include "models/musc.hpp"
#include <torch/script.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <cmath>
#include <set>

namespace musc {
using Weights = std::map<std::string,Tensor>;
static Weights load_visual(const fs::path& file, bool dino, at::Device device) {
    TORCH_CHECK(fs::is_regular_file(file),"Missing visual weights: ",file.u8string());
    Weights tensors;
    const auto dtype=device.is_cuda()?at::kHalf:at::kFloat;
    auto accept = [&](std::string name) {
        return dino || name.rfind("visual.",0)==0;
    };
    auto store = [&](std::string name, Tensor value) {
        if(accept(name)) tensors.emplace(dino?name:name.substr(7),value.to(device,dtype).contiguous());
    };
    if(file.extension()==".safetensors") {
        std::ifstream in(file,std::ios::binary);
        uint64_t length=0;
        in.read(reinterpret_cast<char*>(&length),8);
        TORCH_CHECK(in && length>0 && length<100000000,"Invalid safetensors header");
        std::string header(length,' ');
        in.read(header.data(),length);
        auto table=YAML::Load(header);
        for(auto entry:table) {
            auto name=entry.first.as<std::string>();
            if(name=="__metadata__" || !accept(name)) continue;
            auto info=entry.second;
            auto offsets=info["data_offsets"].as<std::vector<int64_t>>();
            auto shape=info["shape"].as<std::vector<int64_t>>();
            auto type=info["dtype"].as<std::string>();
            TORCH_CHECK(type=="F32" || type=="F16" || type=="BF16","Unsupported tensor dtype ",type);
            Tensor data=at::empty(shape,type=="F32"?at::kFloat:type=="F16"?at::kHalf:at::kBFloat16);
            TORCH_CHECK(offsets.size()==2 && offsets[0]>=0 && offsets[1]-offsets[0]==data.nbytes(),"Invalid offsets ",name);
            in.seekg(8+length+offsets[0]);
            in.read(static_cast<char*>(data.data_ptr()),data.nbytes());
            TORCH_CHECK(in,"Truncated checkpoint ",name);
            store(name,data);
        }
    } else {
        std::ifstream stream(file,std::ios::binary);
        caffe2::serialize::PyTorchStreamReader reader(&stream);
        if(reader.hasRecord("constants.pkl")) {
            auto archive=torch::jit::load(file.u8string(),at::kCPU);
            for(const auto& p:archive.named_parameters()) store(p.name,p.value);
        } else {
            auto state=torch::jit::readArchiveAndTensors("data","","",std::nullopt,std::nullopt,at::Device(at::kCPU),reader).toGenericDict();
            if(state.contains("state_dict")) state=state.at("state_dict").toGenericDict();
            for(const auto& p:state) {
                if(!p.key().isString() || !p.value().isTensor()) continue;
                auto key=p.key().toStringRef();
                if(key.rfind("module.",0)==0) key=key.substr(7);
                store(key,p.value().toTensor());
            }
        }
    }
    return tensors;
}
const Tensor& Vision::get(const std::string& key) const {
    auto iter=weights.find(key);
    TORCH_CHECK(iter!=weights.end(),"Missing checkpoint tensor: ",key);
    return iter->second;
}
Tensor Vision::norm(const Tensor& x,const std::string& key,double eps) const {
    return at::layer_norm(x,{x.size(-1)},get(key+".weight"),get(key+".bias"),eps);
}
Vision::Vision(const Config& c,at::Device target):selected(c.layers),
    dino(c.backbone.rfind("dino",0)==0),v2(c.backbone.rfind("dinov2",0)==0),device(target) {
    weights=load_visual(c.weights,dino,device);
    auto kernel=get(dino?"patch_embed.proj.weight":"conv1.weight");
    stride=static_cast<int>(kernel.size(2));
    heads=static_cast<int>(kernel.size(0))/64;
    grid=c.size/stride;
    TORCH_CHECK(grid>0,"Image is smaller than one patch");
    auto embed=get(dino?"pos_embed":"positional_embedding").reshape({-1,kernel.size(0)});
    int old=static_cast<int>(std::sqrt(embed.size(0)-1));
    TORCH_CHECK(old*old==embed.size(0)-1,"Invalid positional grid");
    quick=!dino && c.pretrained=="openai" && c.size==old*stride;
    if(grid==old && (!dino || c.size==old*stride)) pos=embed;
    else {
        auto spatial=embed.slice(0,1).reshape({1,old,old,-1}).permute({0,3,1,2}).to(at::kFloat);
        std::optional<double> scale=dino?std::optional<double>((grid+0.1)/old):std::nullopt;
        spatial=at::upsample_bicubic2d(spatial,{grid,grid},false,scale,scale);
        pos=at::cat({embed.slice(0,0,1),spatial.flatten(2).transpose(1,2).squeeze(0).to(embed.scalar_type())},0);
    }
    const std::string prefix=dino?"blocks.":"transformer.resblocks.";
    for(depth=0;weights.count(prefix+std::to_string(depth)+(dino?".norm1.weight":".ln_1.weight"));++depth) {}
    TORCH_CHECK(depth>0,"No transformer blocks in checkpoint");
    if(!dino) {
        const std::map<std::string,std::pair<int,int>> expected{{"ViT-L-14-336",{1024,14}},{"ViT-L-14",{1024,14}},{"ViT-B-16",{768,16}},{"ViT-B-32",{768,32}},{"ViT-B-16-plus-240",{896,16}}};
        auto it=expected.find(c.backbone);
        TORCH_CHECK(it!=expected.end() && it->second.first==kernel.size(0) && it->second.second==stride,"Backbone name and checkpoint architecture disagree");
    }
    for(int layer:selected) TORCH_CHECK(layer>=0 && layer<depth,"Invalid feature layer ",layer," for depth ",depth);
    if(dino && !v2) {
        int offset=depth-1-*std::max_element(selected.begin(),selected.end());
        for(int& layer:selected) layer+=offset;
    }
}
Features Vision::operator()(Tensor image) const {
    auto kernel=get(dino?"patch_embed.proj.weight":"conv1.weight");
    image=image.to(device,kernel.scalar_type());
    auto bias=dino?std::optional<Tensor>(get("patch_embed.proj.bias")):std::nullopt;
    auto x=at::conv2d(image,kernel,bias,{stride,stride}).flatten(2).transpose(1,2);
    auto cls=get(dino?"cls_token":"class_embedding").reshape({1,1,-1}).expand({x.size(0),1,-1});
    x=at::cat({cls,x},1)+pos;
    if(!dino) x=norm(x,"ln_pre",1e-5);
    Features out;
    auto linear=[&](const Tensor& data,const std::string& key){return at::linear(data,get(key+".weight"),get(key+".bias"));};
    std::map<int,Tensor> chosen;
    for(int i=0;i<depth;++i) {
        auto prefix=std::string(dino?"blocks.":"transformer.resblocks.")+std::to_string(i)+".";
        auto y=norm(x,prefix+(dino?"norm1":"ln_1"),dino?1e-6:1e-5);
        auto qkv=dino?linear(y,prefix+"attn.qkv"):at::linear(y,get(prefix+"attn.in_proj_weight"),get(prefix+"attn.in_proj_bias"));
        qkv=qkv.reshape({x.size(0),x.size(1),3,heads,x.size(2)/heads}).permute({2,0,3,1,4});
        y=at::scaled_dot_product_attention(qkv[0],qkv[1],qkv[2]).transpose(1,2).reshape_as(x);
        y=linear(y,prefix+(dino?"attn.proj":"attn.out_proj"));
        if(v2) y=y*get(prefix+"ls1.gamma");
        x=x+y;
        y=norm(x,prefix+(dino?"norm2":"ln_2"),dino?1e-6:1e-5);
        y=linear(y,prefix+(dino?"mlp.fc1":"mlp.c_fc"));
        y=quick?y*at::sigmoid(1.702*y):at::gelu(y);
        y=linear(y,prefix+(dino?"mlp.fc2":"mlp.c_proj"));
        if(v2) y=y*get(prefix+"ls2.gamma");
        x=x+y;
        if(std::find(selected.begin(),selected.end(),i)!=selected.end()) {
            auto tokens=dino?norm(x,"norm",1e-6):x;
            chosen[i]=tokens.slice(1,1).to(at::kCPU);
        }
    }
    for(int layer:selected) out.patches.push_back(chosen.at(layer));
    auto global=dino?norm(x,"norm",1e-6).select(1,0):at::matmul(norm(x.select(1,0),"ln_post",1e-5),get("proj"));
    global=global.to(at::kFloat);
    if(!dino) global=global/global.norm(2,-1,true).clamp_min(1e-12);
    out.global=global.cpu();
    return out;
}
}
