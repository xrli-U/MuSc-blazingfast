#pragma once
#include <ATen/ATen.h>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace musc {
namespace fs = std::filesystem;
using at::Tensor;
struct Config {
    fs::path data, weights, output;
    std::string dataset, backbone, pretrained, device, vis_type;
    std::vector<std::string> categories;
    int size, batch, divisions;
    bool vis, excel;
    std::vector<int> layers, radii;
};
struct Sample {
    fs::path image, mask;
    std::string type;
    bool abnormal;
};
using Groups = std::map<std::string, std::vector<Sample>>;
struct Features {
    Tensor global;
    std::vector<Tensor> patches;
};
struct Metrics { double auc = 0, f1 = 0, ap = 0; };
struct Result { Metrics image, pixel; double pro = 0; };
Config configuration(const std::vector<std::string>& args);
Groups read_dataset(const Config&);
void prepare_visa(const fs::path&);
std::vector<std::vector<Sample>> split_data(const std::vector<Sample>&, int);
Tensor read_image(const fs::path&, int, bool);
Tensor read_mask(const Sample&, int, bool);
void heatmap(const fs::path&, const Tensor&, double, double);
class Vision {
    std::map<std::string, Tensor> weights;
    std::vector<int> selected;
    bool dino, v2, quick;
    int depth, stride, heads, grid;
    Tensor pos;
    at::Device device;
    const Tensor& get(const std::string&) const;
    Tensor norm(const Tensor&, const std::string&, double) const;
public:
    Vision(const Config&, at::Device);
    Features operator()(Tensor) const;
};
Tensor aggregate(const Tensor&, int);
Tensor mutual(const Tensor&, at::Device);
Tensor rescore(const Tensor&, const Tensor&, const std::vector<int>&);
Result evaluate(const std::vector<Sample>&, const Tensor&, const Tensor&, bool);
void workbook(const fs::path&, const std::map<std::string, Result>&);
void run(const Config&);
void rescore_file(const fs::path&,const fs::path&,bool);
}
