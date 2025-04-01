# Video-Depth-Anything for Mac

## 项目概述

Video-Depth-Anything 是一个基于深度学习的视频深度估计项目，可以从单目视频中提取深度信息。本文档特别针对在 Mac 设备上运行该项目时的注意事项和修复方案。

## Mac 平台适配修复

在 Mac 平台（特别是使用 Apple Silicon 芯片的设备）上运行该项目时，我们遇到并解决了以下问题：

### 1. 注意力机制兼容性问题

项目中使用了 `xformers` 库来加速注意力计算，但在 Mac 上（尤其是使用 MPS 设备而非 CUDA）时会出现不兼容问题。我们修复了代码以支持在没有 `xformers` 或不使用 CUDA 的情况下平稳运行：

- 添加了非 `xformers` 路径的正确实现
- 确保张量维度在不同计算路径间保持一致性

### 2. 张量维度处理

修复了多个与张量维度处理相关的问题：

- 改进了 `reshape_heads_to_batch_dim` 方法，使其能够处理 4D 张量输入
- 统一了 `xformers` 和非 `xformers` 路径的张量处理逻辑
- 确保了注意力计算后输出的张量能够与后续线性层兼容

### 3. MPS 后端支持

确保在使用 Metal Performance Shaders (MPS) 后端时，模型能够正确运行，并处理了与 CUDA 不同的特殊情况。

## 使用指南

### 环境配置

在 Mac 上运行此项目，推荐以下配置：

1. 使用 Python 3.9+ 和 PyTorch 2.0+
2. 确保 PyTorch 版本支持 MPS（Metal Performance Shaders）加速

### 运行命令

```bash
python3 run.py --input_video ./assets/example_videos/your_video.mp4 --output_dir ./outputs --encoder vits --fp32
```

参数说明：
- `--input_video`: 输入视频路径
- `--output_dir`: 输出结果保存路径
- `--encoder`: 编码器类型，推荐使用 `vits`
- `--fp32`: 在 Mac 上推荐使用 float32 精度，以避免潜在的兼容性问题

### 注意事项

1. 在 Mac 上，`xformers` 路径会自动被禁用，改为使用标准的 PyTorch 实现
2. 处理高分辨率视频时可能会较慢，建议先测试较低分辨率的视频
3. 首次运行时会下载预训练模型，请确保网络连接正常

## 已知问题

1. 在 MPS 设备上，某些复杂的注意力操作可能仍有性能瓶颈
2. 对于非常长的视频，可能需要分段处理以避免内存不足

## 项目修改记录

主要修改包括：

1. `attention.py`: 
   - 改进了 `_attention` 方法以支持 4D 张量输入
   - 增强了 `reshape_heads_to_batch_dim` 方法兼容性

2. `motion_module.py`: 
   - 优化了 `TemporalAttention.forward` 方法中的张量处理逻辑
   - 统一了 xformers 和非 xformers 路径的实现

这些修改确保了项目在 Mac 平台上能够正常运行，同时保持了与其他平台的兼容性。

## 具体修改内容详情

### 1. 在 `attention.py` 文件中：
- **修改了 `_attention` 方法**：重构了该方法以正确处理4D张量输入，而不是将其转换为3D再处理。这样可以直接使用 `F.scaled_dot_product_attention` 函数处理4D输入，提高了效率和兼容性。

- **增强了 `reshape_heads_to_batch_dim` 方法**：添加了对4D张量的支持，检测输入张量的维度并相应地处理，解决了 `ValueError: too many values to unpack (expected 3)` 错误。

### 2. 在 `motion_module.py` 文件中：
- **重构了 `TemporalAttention.forward` 方法**：
  - 分离并优化了 xformers 和非 xformers 路径的逻辑
  - 在 xformers 路径中，添加了将4D输出转换为适合线性层的逻辑
  - 对于非 xformers 路径，恢复使用 `reshape_heads_to_batch_dim` 和 `reshape_batch_dim_to_heads` 处理张量转换

这些修改解决了在 Mac 上（特别是使用 MPS 后端而非 CUDA）运行时的形状不匹配和维度错误问题。主要是确保在没有 xformers 支持的情况下，张量形状在整个处理流程中保持一致。

### 3. 在 `utils/dc_utils.py` 文件中：
- **完全重写了 `save_video` 函数**：尤其针对 Mac 平台采用了特殊处理方案：
  - 首选使用 `imageio.mimwrite()` 方法一次性写入所有影帧，而不是逐帧添加
  - 添加了异常处理来提供替代方案
  - 确保始终指定 `codec='libx264'` 参数以避免 "expected bytes, NoneType found" 错误
  
  这解决了一系列与 PyAVPlugin 相关的兼容性问题：
  ```
  TypeError: PyAVPlugin.write() got an unexpected keyword argument 'macro_block_size'
  TypeError: PyAVPlugin.write() got an unexpected keyword argument 'ffmpeg_params'
  TypeError: expected bytes, NoneType found
  ```
