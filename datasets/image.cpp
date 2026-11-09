#include "models/musc.hpp"
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

namespace musc {
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr,const fs::path& file) {
    TORCH_CHECK(SUCCEEDED(hr),"Image I/O failed: ",file.u8string()," HRESULT ",static_cast<uint32_t>(hr));
}
static ComPtr<IWICImagingFactory> factory() {
    ComPtr<IWICImagingFactory> result;
    check(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&result)),{});
    return result;
}
static Tensor pixels(const fs::path& file,bool gray) {
    auto wic=factory();
    ComPtr<IWICBitmapDecoder> decoder;
    check(wic->CreateDecoderFromFilename(file.c_str(),nullptr,GENERIC_READ,WICDecodeMetadataCacheOnDemand,&decoder),file);
    ComPtr<IWICBitmapFrameDecode> frame;
    check(decoder->GetFrame(0,&frame),file);
    ComPtr<IWICFormatConverter> converted;
    check(wic->CreateFormatConverter(&converted),file);
    check(converted->Initialize(frame.Get(),gray?GUID_WICPixelFormat8bppGray:GUID_WICPixelFormat24bppRGB,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom),file);
    UINT w,h;
    check(converted->GetSize(&w,&h),file);
    int channels=gray?1:3;
    auto buf=at::empty({h,w,channels},at::kByte);
    TORCH_CHECK(buf.nbytes()<=UINT_MAX,"Image too large");
    check(converted->CopyPixels(nullptr,w*channels,static_cast<UINT>(buf.nbytes()),buf.data_ptr<uint8_t>()),file);
    return buf.permute({2,0,1}).unsqueeze(0);
}
struct FilterRow {
    int first;
    std::vector<int> weights;
};
static std::vector<FilterRow> filter_rows(int input,int output,bool cubic) {
    std::vector<FilterRow> rows;
    double scale=double(input)/output, factor=std::max(scale,1.0);
    double support=factor*(cubic?2:1);
    for(int i=0;i<output;++i) {
        double center=(i+0.5)*scale,sum=0;
        int first=std::max(0,int(center-support+0.5)),end=std::min(input,int(center+support+0.5));
        std::vector<double> weights;
        for(int j=first;j<end;++j) {
            double x=std::abs((j+0.5-center)/factor);
            double w=std::max(1.0-x,0.0);
            if(cubic) w=x<1?((1.5*x-2.5)*x*x+1):x<2?(((-0.5*x+2.5)*x-4)*x+2):0;
            weights.push_back(w); sum+=w;
        }
        FilterRow row{first,{}};
        for(double w:weights) row.weights.push_back(int(std::round((w/sum)*(1<<22))));
        rows.push_back(std::move(row));
    }
    return rows;
}
static Tensor resize_bytes(Tensor x,int side,bool cubic) {
    for(int axis:{3,2}) {
        if(x.size(axis)==side) continue;
        auto src=x.transpose(axis,3).contiguous();
        auto length=static_cast<int>(src.size(3));
        auto shape=src.sizes().vec(); shape.back()=side;
        auto dst=at::empty(shape,at::kByte);
        auto weights=filter_rows(length,side,cubic);
        auto from=src.data_ptr<uint8_t>(); auto to=dst.data_ptr<uint8_t>();
        for(int64_t line=0;line<src.numel()/length;++line) {
            for(int i=0;i<side;++i) {
                const auto& row=weights[i];
                int64_t sum=1<<21;
                for(size_t j=0;j<row.weights.size();++j)
                    sum+=int64_t(from[line*length+row.first+j])*row.weights[j];
                to[line*side+i]=static_cast<uint8_t>(std::clamp<int64_t>(sum>>22,0,255));
            }
        }
        x=dst.transpose(axis,3);
    }
    return x;
}
Tensor read_image(const fs::path& file,int side,bool dino) {
    auto x=pixels(file,false);
    if(x.size(2)!=side || x.size(3)!=side) {
        x=resize_bytes(x,side,!dino);
    }
    auto mean=dino?at::tensor({0.485f,0.456f,0.406f}):at::tensor({0.48145466f,0.4578275f,0.40821073f});
    auto sd=dino?at::tensor({0.229f,0.224f,0.225f}):at::tensor({0.26862954f,0.26130258f,0.27577711f});
    return (x[0].to(at::kFloat)/255-mean.view({3,1,1}))/sd.view({3,1,1});
}
Tensor read_mask(const Sample& s,int side,bool visa) {
    if(s.mask.empty()) return at::zeros({side,side},at::kByte);
    auto x=pixels(s.mask,true);
    if(x.size(2)!=side || x.size(3)!=side) x=resize_bytes(x,side,false);
    return x.reshape({side,side}).gt(visa?0:254).to(at::kByte);
}
void heatmap(const fs::path& file,const Tensor& map,double lo,double hi) {
    fs::create_directories(file.parent_path());
    auto x=((map-lo)/std::max(hi-lo,1e-12)).clamp(0,1);
    std::vector<Tensor> channels;
    for(int center:{1,2,3}) channels.push_back((1.5-(4*x-center).abs()).clamp(0,1)*255);
    auto bgr=at::stack(channels,-1).round().to(at::kByte).contiguous();
    auto wic=factory();
    ComPtr<IWICStream> stream;
    check(wic->CreateStream(&stream),file);
    check(stream->InitializeFromFilename(file.c_str(),GENERIC_WRITE),file);
    ComPtr<IWICBitmapEncoder> encoder;
    check(wic->CreateEncoder(GUID_ContainerFormatPng,nullptr,&encoder),file);
    check(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache),file);
    ComPtr<IWICBitmapFrameEncode> frame;
    check(encoder->CreateNewFrame(&frame,nullptr),file);
    check(frame->Initialize(nullptr),file);
    UINT w=static_cast<UINT>(x.size(1)),h=static_cast<UINT>(x.size(0));
    check(frame->SetSize(w,h),file);
    auto format=GUID_WICPixelFormat24bppBGR;
    check(frame->SetPixelFormat(&format),file);
    check(frame->WritePixels(h,w*3,w*h*3,bgr.data_ptr<uint8_t>()),file);
    check(frame->Commit(),file); check(encoder->Commit(),file);
}
}
