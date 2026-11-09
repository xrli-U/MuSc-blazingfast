#include "models/musc.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <set>

namespace musc {
static bool included(const Config& c,const std::string& name) {
    return std::any_of(c.categories.begin(),c.categories.end(),[&](const auto& category){
        return category=="ALL" || category=="all" || category==name;
    });
}
static std::vector<fs::path> images(const fs::path& dir) {
    std::vector<fs::path> result;
    for(auto& entry:fs::directory_iterator(dir)) {
        auto ext=entry.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        static const std::set<std::string> known{".png",".bmp",".jpg",".jpeg",".tiff",".tif"};
        if(entry.is_regular_file() && known.count(ext)) result.push_back(entry.path());
    }
    std::sort(result.begin(),result.end());
    return result;
}
static std::vector<std::string> csvrow(const std::string& line) {
    std::vector<std::string> cells(1);
    bool quoted=false;
    for(size_t i=0;i<line.size();++i) {
        char ch=line[i];
        if(ch=='"' && quoted && i+1<line.size() && line[i+1]=='"') { cells.back()+='"'; ++i; }
        else if(ch=='"') quoted=!quoted;
        else if(ch==',' && !quoted) cells.emplace_back();
        else if(ch!='\r') cells.back()+=ch;
    }
    return cells;
}
void prepare_visa(const fs::path& root) {
    std::ifstream csv(root/"split_csv"/"1cls.csv");
    TORCH_CHECK(csv,"VisA split_csv/1cls.csv not found");
    using Rows = std::vector<std::vector<std::string>>;
    std::map<std::string,std::map<std::string,Rows>> groups;
    std::string line;
    std::getline(csv,line);
    while(std::getline(csv,line)) {
        if(line.empty()) continue;
        auto fields=csvrow(line);
        TORCH_CHECK(fields.size()>=5,"Invalid VisA CSV row");
        groups[fields[1]][fields[0]].push_back(fields);
    }
    YAML::Emitter json;
    json << YAML::Flow << YAML::DoubleQuoted << YAML::BeginMap;
    for(auto& [phase,classes]:groups) {
        json << YAML::Key << phase << YAML::Value << YAML::BeginMap;
        for(auto& [name,rows]:classes) {
            json << YAML::Key << name << YAML::Value << YAML::BeginSeq;
            for(auto& row:rows) {
                bool abnormal=row[2]=="anomaly";
                json << YAML::BeginMap << YAML::Key << "img_path" << YAML::Value << row[3]
                     << YAML::Key << "mask_path" << YAML::Value << (abnormal?row[4]:"")
                     << YAML::Key << "cls_name" << YAML::Value << name
                     << YAML::Key << "specie_name" << YAML::Value << ""
                     << YAML::Key << "anomaly" << YAML::Value << int(abnormal) << YAML::EndMap;
            }
            json << YAML::EndSeq;
        }
        json << YAML::EndMap;
    }
    json << YAML::EndMap;
    TORCH_CHECK(json.good(),"Could not encode VisA metadata");
    auto temp=root/"meta.json.tmp";
    TORCH_CHECK(!fs::exists(temp),"Temporary metadata file already exists");
    std::ofstream out(temp,std::ios::binary);
    out << json.c_str() << '\n'; out.close();
    TORCH_CHECK(out,"Failed writing VisA metadata");
    fs::rename(temp,root/"meta.json");
}
Groups read_dataset(const Config& c) {
    Groups groups;
    TORCH_CHECK(fs::is_directory(c.data),"Dataset directory not found: ",c.data.u8string());
    if(c.dataset=="visa") {
        if(fs::exists(c.data/"meta.json")) {
            std::ifstream input(c.data/"meta.json");
            auto test=YAML::Load(input)["test"];
            for(auto pair:test) {
                auto name=pair.first.as<std::string>();
                if(!included(c,name)) continue;
                for(auto row:pair.second) {
                    auto image=row["img_path"].as<std::string>(),mask=row["mask_path"].as<std::string>();
                    bool anomaly=row["anomaly"]?row["anomaly"].as<int>()!=0:image.find("Normal")==std::string::npos;
                    groups[name].push_back({c.data/fs::u8path(image),mask.empty()?fs::path{}:c.data/fs::u8path(mask),anomaly?"Anomaly":"Normal",anomaly});
                }
            }
        } else {
            std::ifstream file(c.data/"split_csv"/"1cls.csv");
            TORCH_CHECK(file,"VisA requires meta.json or split_csv/1cls.csv");
            std::string line;
            std::getline(file,line);
            while(std::getline(file,line)) {
                auto row=csvrow(line);
                if(row.size()<5 || row[1]!="test" || !included(c,row[0])) continue;
                bool abnormal=row[2]=="anomaly";
                groups[row[0]].push_back({c.data/fs::u8path(row[3]),abnormal?c.data/fs::u8path(row[4]):fs::path{},abnormal?"Anomaly":"Normal",abnormal});
            }
        }
    } else {
        for(auto& category:fs::directory_iterator(c.data)) {
            auto name=category.path().filename().u8string();
            auto test=category.path()/"test";
            if(!included(c,name) || !fs::is_directory(test)) continue;
            std::map<std::string,fs::path> types;
            for(auto& dir:fs::directory_iterator(test)) if(dir.is_directory()) types[dir.path().filename().u8string()]=dir.path();
            for(auto& [type,dir]:types) {
                bool abnormal=type!=(c.dataset=="btad"?"ok":"good");
                auto files=images(dir);
                std::vector<fs::path> masks;
                if(abnormal) masks=images(category.path()/"ground_truth"/fs::u8path(type));
                TORCH_CHECK(!abnormal || files.size()==masks.size(),"Image/mask count mismatch: ",dir.u8string());
                for(size_t i=0;i<files.size();++i) groups[name].push_back({files[i],abnormal?masks[i]:fs::path{},type,abnormal});
            }
        }
    }
    TORCH_CHECK(!groups.empty(),"No matching test categories in ",c.data.u8string());
    return groups;
}
class Shuffle42 {
    uint32_t state[624];
    int cursor=624;
    uint32_t draw() {
        if(cursor==624) {
            for(int i=0;i<624;++i) {
                uint32_t x=(state[i]&0x80000000u)|(state[(i+1)%624]&0x7fffffffu);
                state[i]=state[(i+397)%624]^(x>>1)^((x&1)?0x9908b0dfu:0);
            }
            cursor=0;
        }
        auto x=state[cursor++];
        x^=x>>11; x^=(x<<7)&0x9d2c5680; x^=(x<<15)&0xefc60000;
        return x^(x>>18);
    }
public:
    Shuffle42() {
        state[0]=19650218;
        for(int i=1;i<624;++i) state[i]=1812433253u*(state[i-1]^(state[i-1]>>30))+i;
        int i=1;
        for(int k=0;k<624;++k) {
            state[i]=(state[i]^((state[i-1]^(state[i-1]>>30))*1664525u))+42;
            if(++i==624) {state[0]=state[623];i=1;}
        }
        for(int k=0;k<623;++k) {
            state[i]=(state[i]^((state[i-1]^(state[i-1]>>30))*1566083941u))-i;
            if(++i==624) {state[0]=state[623];i=1;}
        }
        state[0]=0x80000000u;
    }
    size_t below(size_t n) {
        int bits=0; for(auto m=n;m;m>>=1) ++bits;
        uint32_t v; do {v=draw()>>(32-bits);} while(v>=n);
        return v;
    }
};
std::vector<std::vector<Sample>> split_data(const std::vector<Sample>& input,int n) {
    if(n==1) return {input};
    std::vector<std::string> order;
    std::map<std::string,std::vector<Sample>> types;
    for(auto& s:input) {
        if(!types.count(s.type)) order.push_back(s.type);
        types[s.type].push_back(s);
    }
    Shuffle42 random;
    std::vector<std::vector<Sample>> parts(n);
    for(auto& type:order) {
        auto& group=types[type];
        for(size_t i=group.size();i>1;--i) std::swap(group[i-1],group[random.below(i)]);
        for(size_t i=0;i<group.size();++i) parts[i%n].push_back(group[i]);
    }
    return parts;
}
}
