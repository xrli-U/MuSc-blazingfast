#include "models/musc.hpp"
#include <windows.h>
#include <objbase.h>
#include <iostream>
int wmain(int argc, wchar_t** argv) {
    try {
        std::vector<std::string> args;
        for(int i=1;i<argc;++i) args.push_back(std::filesystem::path(argv[i]).u8string());
        if(std::find(args.begin(),args.end(),"--help")!=args.end()) {
            std::cout<<"musc --data_path PATH [--dataset_name mvtec_ad|visa|btad] [--class_name ALL] [--weights FILE] [--device cpu|0] [--config configs/musc.yaml]\n";
            return 0;
        }
        if(!args.empty() && args[0]=="--prepare-visa") {
            TORCH_CHECK(args.size()==2,"Usage: musc --prepare-visa DATASET_ROOT");
            musc::prepare_visa(std::filesystem::u8path(args[1]));
            return 0;
        }
        if(!args.empty() && args[0]=="--rescore") {
            TORCH_CHECK(args.size()==4 && (args[3]=="mvtec_ad" || args[3]=="visa"),"Usage: musc --rescore features.bin scores.txt mvtec_ad|visa");
            musc::rescore_file(std::filesystem::u8path(args[1]),std::filesystem::u8path(args[2]),args[3]=="visa");
            return 0;
        }
        auto hr=CoInitializeEx(nullptr,COINIT_MULTITHREADED);
        if(FAILED(hr)) throw std::runtime_error("Windows Imaging Component initialization failed");
        at::InferenceMode guard;
        musc::run(musc::configuration(args));
        CoUninitialize();
        return 0;
    } catch(const c10::Error& e) { std::cerr<<e.what_without_backtrace()<<'\n'; }
      catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; }
    return 1;
}
