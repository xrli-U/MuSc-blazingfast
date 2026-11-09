# MuSc C++

MuSc detects defects by comparing visual patches across unlabeled test images of the same product. This Windows x64 implementation runs the vision encoder, mutual scoring, image rescoring and evaluation in C++17 with LibTorch. It produces one anomaly score per image and a pixel-level anomaly map. Ground-truth masks are used for evaluation. [Paper (ICLR 2024)](https://arxiv.org/abs/2401.16753) · [Original MuSc implementation](https://github.com/xrli-U/MuSc)

## Build on Windows

Install Visual Studio 2022 or Build Tools 2022 with **Desktop development with C++**, an x64 MSVC toolchain and a Windows SDK. Use CMake 3.24 or newer. Run the commands below in PowerShell from the project root.

1. Download and extract a **Release, shared-with-deps** Windows x64 LibTorch 2.4.0 package. For GPU inference, use the CUDA 12.1 package and a compatible NVIDIA driver. [CUDA 12.1 package](https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.4.0%2Bcu121.zip) · [CPU package](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%2Bcpu.zip)
2. Set `LIBTORCH_ROOT` to the extracted directory containing `include/` and `lib/torch_cpu.lib`. CMake also checks `third_party/libtorch` and `%LOCALAPPDATA%/MuSc/deps/libtorch`. The first configure downloads a checksum-pinned copy of yaml-cpp 0.8.0 if no compatible installation is found. For an offline build, pass `-Dyaml-cpp_DIR=...` pointing to the directory containing `yaml-cpp-config.cmake`. [yaml-cpp 0.8.0](https://github.com/jbeder/yaml-cpp/releases/tag/0.8.0)

```powershell
$env:LIBTORCH_ROOT = 'D:/dependencies/libtorch'
cmake --preset windows-x64
cmake --build --preset release
.\build\Release\musc.exe --help
```

3. Fetch the default OpenAI CLIP ViT-L/14@336px checkpoint. The script verifies SHA-256 and reuses an existing valid file. The checkpoint is stored in `models/backbone/weights/ViT-L-14-336px.pt`. You can also pass a local checkpoint with `--weights PATH`. [OpenAI CLIP](https://github.com/openai/CLIP)

```powershell
.\scripts\fetch_weights.ps1
```

## Run a first evaluation

Set `$data` to the extracted MVTec AD root, the directory containing `bottle/`. In each category, normal test images go in `test/good/`. Anomalous images go in `test/<defect>/` with corresponding masks in `ground_truth/<defect>/`. The loader pairs images and masks in sorted filename order within each defect. [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

```powershell
$data = 'D:/datasets/mvtec_anomaly_detection'
.\build\Release\musc.exe --dataset_name mvtec_ad --data_path "$data" --class_name bottle --device 0 --save_excel true
```

Use `--device cpu` for CPU inference. With `--device 0` the program requests the first CUDA device and falls back to CPU if CUDA is unavailable. Check the `device=` line at startup. The example writes the following files under `output/mvtec_ad/ViT-L-14-336/imagesize518/`.

| Output | Contents |
| --- | --- |
| `bottle/scores.csv` | Input image path and anomaly score, one row per image. Higher scores indicate stronger evidence of a defect. |
| `results.xlsx` | Image AUROC, maximum F1 and AP, plus pixel AUROC, maximum F1, AP and AUPRO. The console reports these metrics as percentages. |
| `bottle/<defect>/*.png` | Optional heatmaps with `--vis true`. The default `single_norm` scales each abnormal image separately. Use `--vis_type whole_norm` for all images on a common scale within the category. |

## Configure an experiment

Defaults are in [configs/musc.yaml](configs/musc.yaml). Command-line values override the configuration. The default run uses 518 × 518 images, a batch of 4, feature layers 5, 11, 17 and 23 (zero-based), and neighborhood widths 1, 3 and 5.

| Option | Usage |
| --- | --- |
| `--class_name` | Use one category, a list such as `bottle capsule`, or `ALL` (default). |
| `--batch_size`, `--divide_num` | Lower the batch size to reduce encoder memory use. Increasing the number of subsets also reduces the mutual-scoring reference set and can change scores. Each subset needs at least two images. |
| `--output_dir`, `--config` | Choose the output root or load a different YAML file. Relative paths resolve from the working directory. |
| `--backbone_name`, `--pretrained`, `--weights` | For another supported ViT backbone, supply matching local weights and feature layers. The default OpenAI encoder reads TorchScript. The loader also accepts compatible OpenCLIP or DINO safetensors with matching tensor names and dimensions. |

**Other datasets.** For BTAD, use `--dataset_name btad` and the same category layout with `test/ok/` for normal images. For VisA, use `--dataset_name visa` and supply its root with `split_csv/1cls.csv` or `meta.json`. The optional command `musc.exe --prepare-visa DATASET_ROOT` creates metadata from the split CSV.

**Standalone RsCIN.** Use `musc.exe --rescore features.bin scores.txt mvtec_ad` (or `visa`) to rescore existing predictions. The feature file starts with two little-endian int64 values (image count and feature width), followed by a row-major float32 matrix. Provide one input score per image in matching order. The command prints rescored values to standard output. Normal evaluation extracts its own features.

**VS Code.** Install the C/C++ and CMake Tools extensions, open the project folder and set the dataset path and category in `configs/musc.yaml`. Use **Ctrl+Shift+B** to build or **Ctrl+F5** to build and run the supplied launch configuration. If CMake is outside your PATH, set `cmake.cmakePath` in your VS Code user settings.

## License

The MuSc source code is released under the [MIT License](LICENSE). Obtain model weights and datasets from their respective providers under the terms they publish.
