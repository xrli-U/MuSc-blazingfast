# MuSc C++

MuSc 通过比较同一产品多张未标注测试图像的局部特征来发现缺陷。本项目面向 Windows x64，使用 C++17 和 LibTorch 完成视觉特征提取、图块互评分、图像重评分及评估。每张图像会得到一个异常分数和一张像素级异常图，真值掩码用于计算评估指标。[论文（ICLR 2024）](https://arxiv.org/abs/2401.16753) · [MuSc 原始实现](https://github.com/xrli-U/MuSc)

## 在 Windows 上构建

安装 Visual Studio 2022 或 Build Tools 2022，选择 **使用 C++ 的桌面开发**、x64 MSVC 工具链和 Windows SDK，并准备 CMake 3.24 或更高版本。以下 PowerShell 命令均在项目根目录执行。

1. 下载并解压 Windows x64 的 LibTorch 2.4.0，选择 **Release、shared-with-deps** 发行包。使用 NVIDIA GPU 时，选择 CUDA 12.1 发行包并安装兼容驱动。[CUDA 12.1 版](https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.4.0%2Bcu121.zip) · [CPU 版](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%2Bcpu.zip)
2. 将 `LIBTORCH_ROOT` 指向包含 `include/` 和 `lib/torch_cpu.lib` 的解压目录。CMake 也会查找 `third_party/libtorch` 和 `%LOCALAPPDATA%/MuSc/deps/libtorch`。如果机器上没有兼容的 yaml-cpp，首次配置会下载 0.8.0 源码并校验哈希。离线构建可用 `-Dyaml-cpp_DIR=...` 指定 `yaml-cpp-config.cmake` 所在目录。[yaml-cpp 0.8.0](https://github.com/jbeder/yaml-cpp/releases/tag/0.8.0)

```powershell
$env:LIBTORCH_ROOT = 'D:/dependencies/libtorch'
cmake --preset windows-x64
cmake --build --preset release
.\build\Release\musc.exe --help
```

3. 获取默认的 OpenAI CLIP ViT-L/14@336px 权重。脚本会校验 SHA-256，并复用已有的有效文件。默认位置是 `models/backbone/weights/ViT-L-14-336px.pt`，也可用 `--weights PATH` 指定本地权重。[OpenAI CLIP](https://github.com/openai/CLIP)

```powershell
.\scripts\fetch_weights.ps1
```

## 运行第一次评估

将 `$data` 替换为 MVTec AD 解压后包含 `bottle/` 的目录。每个类别的正常图放在 `test/good/`，异常图放在 `test/<缺陷类型>/`，对应掩码放在 `ground_truth/<缺陷类型>/`。程序在每个缺陷类型内按文件名排序配对图像与掩码。[MVTec AD 数据集](https://www.mvtec.com/company/research/datasets/mvtec-ad)

```powershell
$data = 'D:/datasets/mvtec_anomaly_detection'
.\build\Release\musc.exe --dataset_name mvtec_ad --data_path "$data" --class_name bottle --device 0 --save_excel true
```

用 `--device cpu` 可在 CPU 上运行。`--device 0` 请求第一块 CUDA 显卡，CUDA 不可用时回退到 CPU，可在启动时查看 `device=` 确认。上述命令在 `output/mvtec_ad/ViT-L-14-336/imagesize518/` 下生成以下结果。

| 输出 | 内容 |
| --- | --- |
| `bottle/scores.csv` | 每张图一行，记录输入路径与异常分数。分数越高，图像异常的证据越强。 |
| `results.xlsx` | 图像级 AUROC、最大 F1、AP，以及像素级 AUROC、最大 F1、AP、AUPRO。终端也会以百分数打印这七项指标。 |
| `bottle/<缺陷类型>/*.png` | 使用 `--vis true` 保存热图。默认的 `single_norm` 对每张异常图单独归一化。使用 `--vis_type whole_norm` 可按类别统一色标，保存全部图像的热图。 |

## 配置实验

默认配置见 [configs/musc.yaml](configs/musc.yaml)，命令行参数优先。默认输入为 518 × 518，特征提取批量为 4，提取第 5、11、17、23 层（从 0 开始编号），邻域宽度为 1、3、5。

| 参数 | 用法 |
| --- | --- |
| `--class_name` | 可指定单个类别、用空格分开的多个类别（如 `bottle capsule`），或默认值 `ALL`。 |
| `--batch_size`、`--divide_num` | 减小批量可降低编码阶段的内存占用。增大子集数量还会缩小互评分的参考集，可能改变分数。每个子集至少要有两张测试图。 |
| `--output_dir`、`--config` | 可指定输出根目录或另一个 YAML 配置文件。相对路径以程序的当前工作目录为准。 |
| `--backbone_name`、`--pretrained`、`--weights` | 切换到其他受支持的 ViT 骨干时，应提供匹配的本地权重并设置特征层。默认 OpenAI 编码器读取 TorchScript，加载器也支持张量名称和维度匹配的 OpenCLIP 或 DINO safetensors。 |

**其他数据集。** BTAD 使用 `--dataset_name btad`，类别目录结构相同，正常图放在 `test/ok/`。VisA 使用 `--dataset_name visa`，数据集根目录需有 `split_csv/1cls.csv` 或 `meta.json`。可运行 `musc.exe --prepare-visa DATASET_ROOT` 从划分文件生成元数据。

**独立使用 RsCIN。** 运行 `musc.exe --rescore features.bin scores.txt mvtec_ad`（或将末项换成 `visa`）可对已有预测重新评分。特征文件的前两个小端 int64 值为图像数与特征维度，随后是按行存放的 float32 矩阵。分数文本每行一个值，顺序与特征行一致。新分数输出到终端。常规评估会自行提取所需特征。

**VS Code。** 安装 C/C++ 和 CMake Tools 扩展，打开项目文件夹，在 `configs/musc.yaml` 中设置数据集路径和类别。按 **Ctrl+Shift+B** 构建，或按 **Ctrl+F5** 使用仓库提供的启动配置进行构建和运行。如果 CMake 未加入 PATH，可在 VS Code 用户设置中填写 `cmake.cmakePath`。

## 许可证

MuSc 源码采用 [MIT 许可证](LICENSE)。模型权重与数据集由各自提供方分发，使用时请遵循其公布的条款。
