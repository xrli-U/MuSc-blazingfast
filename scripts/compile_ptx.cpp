#include <windows.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

template<class Function>
Function symbol(HMODULE dll, const char* name) {
    auto entry = GetProcAddress(dll, name);
    if (!entry) throw std::runtime_error(std::string("Missing NVRTC function ") + name);
    return reinterpret_cast<Function>(entry);
}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Expected NVRTC, builtins, kernel source and output paths");
        if (!LoadLibraryW(argv[2])) throw std::runtime_error("Failed to load NVRTC builtins");
        auto dll = LoadLibraryW(argv[1]);
        if (!dll) throw std::runtime_error("Failed to load NVRTC");
        auto create = symbol<int(WINAPI*)(void**, const char*, const char*, int, const char* const*, const char* const*)>(dll, "nvrtcCreateProgram");
        auto compile = symbol<int(WINAPI*)(void*, int, const char* const*)>(dll, "nvrtcCompileProgram");
        using Size = int(WINAPI*)(void*, size_t*);
        using Read = int(WINAPI*)(void*, char*);
        std::ifstream input(std::filesystem::path(argv[3]), std::ios::binary);
        if (!input) throw std::runtime_error("Failed to open CUDA kernel");
        std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        void* program = nullptr;
        if (create(&program, source.c_str(), "_MSM_tiled.cu", 0, nullptr, nullptr))
            throw std::runtime_error("Failed to create NVRTC program");
        const char* options[] = {"--gpu-architecture=compute_75", "--std=c++17"};
        size_t size = 0;
        if (compile(program, 2, options)) {
            symbol<Size>(dll, "nvrtcGetProgramLogSize")(program, &size);
            std::vector<char> log(size + 1);
            symbol<Read>(dll, "nvrtcGetProgramLog")(program, log.data());
            throw std::runtime_error(std::string("CUDA kernel compilation failed: ") + log.data());
        }
        if (symbol<Size>(dll, "nvrtcGetPTXSize")(program, &size) || !size)
            throw std::runtime_error("Failed to get PTX size");
        std::vector<char> ptx(size);
        if (symbol<Read>(dll, "nvrtcGetPTX")(program, ptx.data()))
            throw std::runtime_error("Failed to get PTX");
        symbol<int(WINAPI*)(void**)>(dll, "nvrtcDestroyProgram")(&program);
        auto destination = std::filesystem::path(argv[4]);
        auto temporary = destination;
        temporary += L".tmp";
        std::ofstream output(temporary, std::ios::binary);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << "static constexpr unsigned char tiled_ptx[] = {";
        for (size_t i = 0; i < size; ++i) {
            if (i % 32 == 0) output << '\n';
            output << static_cast<unsigned int>(static_cast<unsigned char>(ptx[i])) << ',';
        }
        output << "\n};\n";
        output.close();
        std::filesystem::rename(temporary, destination);
        return 0;
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        return 1;
    }
}
