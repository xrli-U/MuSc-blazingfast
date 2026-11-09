#include "models/musc.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace musc {
void rescore_file(const fs::path& features,const fs::path& scores,bool visa) {
    std::ifstream data(features,std::ios::binary);
    int64_t n=0,width=0;
    data.read(reinterpret_cast<char*>(&n),8);
    data.read(reinterpret_cast<char*>(&width),8);
    TORCH_CHECK(data && n>0 && width>0 && n<=1000000 && width<=8192 &&
        fs::file_size(features)==16+uint64_t(n)*width*sizeof(float),"Invalid feature matrix");
    auto cls=at::empty({n,width},at::kFloat);
    data.read(static_cast<char*>(cls.data_ptr()),cls.nbytes());
    std::ifstream input(scores);
    TORCH_CHECK(input && data,"Cannot read scores or features");
    std::vector<float> values;
    float val;
    while(input>>val) values.push_back(val);
    TORCH_CHECK(input.eof() && values.size()==n,"Provide one numeric score per line in dataset order");
    auto out=rescore(at::tensor(values),cls,visa?std::vector<int>{1,8,9}:std::vector<int>{1,2,3});
    std::cout<<std::setprecision(9);
    for(int64_t i=0;i<n;++i) std::cout<<out[i].item<float>()<<'\n';
}
}
