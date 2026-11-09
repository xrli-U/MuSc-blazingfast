#include "models/musc.hpp"
#include <yaml-cpp/yaml.h>
#include <set>
namespace musc {
Config configuration(const std::vector<std::string>& args) {
    std::map<std::string, std::vector<std::string>> opts;
    const std::set<std::string> keys{"config","data_path","dataset_name","class_name","device","output_dir","vis","vis_type","save_excel","r_list","feature_layers","backbone_name","pretrained","weights","img_resize","batch_size","divide_num"};
    for (size_t i = 0; i < args.size();) {
        auto key = args[i++];
        TORCH_CHECK(key.rfind("--",0)==0 && keys.count(key.substr(2)), "Unknown option: ", key);
        auto& values = opts[key.substr(2)];
        while (i < args.size() && args[i].rfind("--",0)!=0) values.push_back(args[i++]);
        TORCH_CHECK(!values.empty(), "Value required for ", key);
    }
    auto cfg = YAML::LoadFile(opts.count("config") ? opts.at("config").front() : "configs/musc.yaml");
    auto read = [&](const char* group, const char* key) {
        if(opts.count(key)) return YAML::Node(opts.at(key).at(0));
        return group[0] ? cfg[group][key] : cfg[key];
    };
    auto list = [&](const char* key) {
        if (!opts.count(key)) return cfg["models"][key].as<std::vector<int>>();
        std::vector<int> out;
        for(auto& value : opts.at(key)) out.push_back(std::stoi(value));
        return out;
    };
    Config c;
    c.data = fs::u8path(read("datasets","data_path").as<std::string>());
    c.dataset = read("datasets","dataset_name").as<std::string>();
    auto classes=read("datasets","class_name");
    if(opts.count("class_name")) c.categories=opts.at("class_name");
    else if(classes.IsSequence()) c.categories=classes.as<std::vector<std::string>>();
    else c.categories={classes.as<std::string>()};
    c.size = read("datasets","img_resize").as<int>();
    c.divisions = read("datasets","divide_num").as<int>();
    c.batch = read("models","batch_size").as<int>();
    c.backbone = read("models","backbone_name").as<std::string>();
    c.pretrained = read("models","pretrained").as<std::string>();
    c.layers = list("feature_layers"); c.radii = list("r_list");
    c.device = read("","device").as<std::string>();
    c.output = fs::u8path(read("testing","output_dir").as<std::string>());
    c.vis = read("testing","vis").as<bool>();
    c.excel = read("testing","save_excel").as<bool>();
    c.vis_type = read("testing","vis_type").as<std::string>();
    fs::path default_file = c.backbone=="ViT-L-14-336" && c.pretrained=="openai"
        ? "ViT-L-14-336px.pt" : c.backbone + (c.backbone.rfind("dino",0)==0 ? "" : "-"+c.pretrained) + ".safetensors";
    c.weights = opts.count("weights") ? fs::u8path(opts.at("weights").front()) : fs::path("models/backbone/weights")/default_file;
    TORCH_CHECK(c.size>0 && c.batch>0 && c.divisions>0 && !c.layers.empty() && !c.radii.empty(),"Invalid sizes or empty layer/radius list");
    for(auto r:c.radii) TORCH_CHECK(r>0 && r%2==1,"Neighborhood widths must be positive and odd");
    TORCH_CHECK(c.dataset=="visa" || c.dataset=="mvtec_ad" || c.dataset=="btad","Unknown dataset");
    TORCH_CHECK(c.vis_type=="single_norm" || c.vis_type=="whole_norm","Unknown visualization mode");
    return c;
}
}
