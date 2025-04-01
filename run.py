# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch
import torch.nn as nn

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')

    args = parser.parse_args()

    # Modify device selection: Prioritize MPS, then CUDA, then CPU
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # --- Workaround for potential weight loading issue in motion_modules[0] --- 
    # Check the final linear layer of the first motion module. 
    # The expected dimension depends on the encoder and the layer index (here, index 2 for motion_modules[0]).
    # Weights might be incorrectly loaded with dim 192 from a different config.
    try:
        motion_module_0 = video_depth_anything.head.motion_modules[0]
        output_layer = motion_module_0.temporal_transformer.transformer_blocks[0].attention_blocks[0].to_out[0]
        # Determine the correct expected dimension based on the chosen encoder
        expected_dim = model_configs[args.encoder]['out_channels'][2] # motion_modules[0] corresponds to out_channels[2]
        incorrect_dim = 192 # The dimension often incorrectly loaded

        # Check if the layer dimensions match the expected dim OR the common incorrect dim
        if output_layer.in_features != expected_dim or output_layer.out_features != expected_dim:
            # If it's not the expected dim, but it IS the common incorrect dim, re-initialize
            if output_layer.in_features == incorrect_dim and output_layer.out_features == incorrect_dim:
                print(f"WARNING: Re-initializing motion_modules[0].temporal_transformer.transformer_blocks[0].attention_blocks[0].to_out[0] Layer.")
                print(f"  Expected shape ({expected_dim}, {expected_dim}), found ({output_layer.out_features}, {output_layer.in_features}). Re-initializing to expected.")
                # Re-initialize the layer with the correct expected dimension
                video_depth_anything.head.motion_modules[0].temporal_transformer.transformer_blocks[0].attention_blocks[0].to_out[0] = nn.Linear(
                    expected_dim, 
                    expected_dim, 
                    bias=output_layer.bias is not None
                ).to(DEVICE).to(output_layer.weight.dtype)
                print("  Layer re-initialized.")
            else:
                # If it's neither the expected nor the common incorrect dim, something else is wrong
                print(f"WARNING: Motion module layer has unexpected dimensions ({output_layer.out_features}, {output_layer.in_features}). Expected ({expected_dim}, {expected_dim}). Check model config or weights.")
        # Optional: Add a check for other motion modules if needed
        # motion_module_1 = video_depth_anything.head.motion_modules[1]
        # expected_dim_1 = model_configs[args.encoder]['out_channels'][1]
        # ... similar check ...
    except Exception as e:
        print(f"WARNING: Could not perform workaround check/fix for motion module weights: {e}")
    # --- End Workaround ---

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()
