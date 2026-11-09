#include "models/musc.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace musc {
static uint32_t checksum(const std::string& data) {
    uint32_t crc=~0u;
    for(unsigned char ch:data) {
        crc^=ch;
        for(int bit=0;bit<8;++bit) crc=(crc>>1)^((crc&1)?0xedb88320u:0);
    }
    return ~crc;
}
static void zip(const fs::path& path,const std::map<std::string,std::string>& parts) {
    std::ofstream out(path,std::ios::binary);
    TORCH_CHECK(out,"Cannot open workbook: ",path.u8string());
    auto put=[&](uint32_t v,int bytes){for(int i=0;i<bytes;++i) out.put(static_cast<char>(v>>(8*i)));};
    std::map<std::string,uint32_t> offsets;
    for(const auto& [name,data]:parts) {
        offsets[name]=static_cast<uint32_t>(out.tellp());
        put(0x04034b50,4);put(20,2);put(0,2);put(0,2);put(0,2);put(33,2);
        put(checksum(data),4);put(data.size(),4);put(data.size(),4);put(name.size(),2);put(0,2);out<<name<<data;
    }
    auto start=static_cast<uint32_t>(out.tellp());
    for(const auto& [name,data]:parts) {
        put(0x02014b50,4);put(20,2);put(20,2);put(0,2);put(0,2);put(0,2);put(33,2);
        put(checksum(data),4);put(data.size(),4);put(data.size(),4);put(name.size(),2);put(0,2);put(0,2);
        put(0,2);put(0,2);put(0,4);put(offsets.at(name),4);out<<name;
    }
    auto length=static_cast<uint32_t>(out.tellp())-start;
    put(0x06054b50,4);put(0,2);put(0,2);put(parts.size(),2);put(parts.size(),2);put(length,4);put(start,4);put(0,2);
    TORCH_CHECK(out,"Failed writing workbook");
}
static std::string escaped(const std::string& name) {
    std::string s;
    for(char c:name) {
        if(c=='&') s+="&amp;";
        else if(c=='<') s+="&lt;";
        else if(c=='>') s+="&gt;";
        else s+=c;
    }
    return s;
}
void workbook(const fs::path& path,const std::map<std::string,Result>& results) {
    std::map<std::string,std::string> parts;
    parts["[Content_Types].xml"]=R"(<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>)";
    auto rel=[](const std::string& type,const std::string& target){return "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/"+type+"\" Target=\""+target+"\"/></Relationships>";};
    parts["_rels/.rels"]=rel("officeDocument","xl/workbook.xml");
    parts["xl/_rels/workbook.xml.rels"]=rel("worksheet","worksheets/sheet1.xml");
    parts["xl/workbook.xml"]=R"(<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="MuSc_results" sheetId="1" r:id="rId1"/></sheets></workbook>)";
    std::ostringstream sheet;
    sheet<<std::setprecision(12)<<R"(<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row>)";
    auto cell=[&](const std::string& v){sheet<<"<c t=\"inlineStr\"><is><t>"<<escaped(v)<<"</t></is></c>";};
    for(auto header:{"category","auroc_px","f1_px","ap_px","aupro","auroc_sp","f1_sp","ap_sp"}) cell(header);
    sheet<<"</row>";
    auto row=[&](const std::string& name,const Result& r){
        sheet<<"<row>";cell(name);
        for(double x:{r.pixel.auc,r.pixel.f1,r.pixel.ap,r.pro,r.image.auc,r.image.f1,r.image.ap}) sheet<<"<c><v>"<<100*x<<"</v></c>";
        sheet<<"</row>";
    };
    for(const auto& [name,r]:results) if(name!="mean") row(name,r);
    row("mean",results.at("mean"));
    sheet<<"</sheetData></worksheet>";
    parts["xl/worksheets/sheet1.xml"]=sheet.str();
    zip(path,parts);
}
}
