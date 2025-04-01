# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import matplotlib.cm as cm
import imageio
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1):
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        frames = vid.get_batch(frames_idx).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps

        stride = max(round(original_fps / fps), 1)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (process_length > 0 and frame_count >= process_length):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    # Mac上使用PyAVPlugin运行时需要特殊处理
    import platform
    import os
    import tempfile
    import subprocess
    
    # 准备帧数据
    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        depth_frames = []
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            depth_frames.append(depth_vis)
        frames_to_save = np.stack(depth_frames)
    else:
        frames_to_save = frames
    
    # 检查和修复张量的形状
    # 确保基础形状是(frames, height, width, channels)
    if len(frames_to_save.shape) == 3 and frames_to_save.shape[2] == 3:  # (frames, width, 3) -> (frames, width, 1, 3)
        # 需要添加缺失的维度
        frames_to_save = frames_to_save.reshape(frames_to_save.shape[0], frames_to_save.shape[1], 1, 3)
    
    # 尝试使用多种方法保存
    save_success = False
    
    # 判断是否是Mac系统
    if platform.system() == 'Darwin':  # macOS
        # 方法1: 使用PNG序列保存然后转MP4
        try:
            # 创建临时目录存放PNG文件
            temp_dir = tempfile.mkdtemp()
            print(f"Saving frames to temporary directory: {temp_dir}")
            
            # 将帧保存为PNG
            for i in range(frames_to_save.shape[0]):
                frame = frames_to_save[i]
                imageio.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.png"), frame)
            
            # 使用ffmpeg将PNG转换为MP4
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-framerate', str(fps), 
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                output_video_path
            ]
            print(f"Running ffmpeg: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True)
            
            # 清理临时文件
            for i in range(frames_to_save.shape[0]):
                os.remove(os.path.join(temp_dir, f"frame_{i:04d}.png"))
            os.rmdir(temp_dir)
            
            save_success = True
            print(f"Successfully saved video to {output_video_path}")
            return
        except Exception as e:
            print(f"Warning: Failed to save with ffmpeg, error: {e}")
    
    # 如果前面的方法失败或非Mac系统，尝试使用imageio
    if not save_success:
        try:
            print("Trying to save with imageio's basic writer...")
            # 使用基本参数
            writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264')
        except Exception as e:
            print(f"Warning: Failed to initialize imageio writer with basic parameters, using minimal config. Error: {e}")
            # 如果还是失败，尝试使用最小参数
            writer = imageio.get_writer(output_video_path, fps=fps)
    else:
        # 其他系统保持原有参数
        writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    # 获取已处理好的帧数据
    try:
        if save_success:  # 如果已经通过前面的方法成功保存，就不需要再处理了
            return
            
        # 确保每一帧的形状正确
        print(f"Saving video using writer, frames shape: {frames_to_save.shape}")
        
        for i in range(frames_to_save.shape[0]):
            frame = frames_to_save[i]
            
            # 处理形状问题，确保是正确的RGB图像格式
            if len(frame.shape) == 3 and frame.shape[0] == 3:  # 从 (3, height, width) 转换为 (height, width, 3)
                frame = np.transpose(frame, (1, 2, 0))
            elif len(frame.shape) == 2:  # 从 grayscale (height, width) 添加颜色通道
                frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
                
            # 确保是uint8类型
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                    
            writer.append_data(frame)
    except Exception as e:
        print(f"Error during frame appending: {e}")

    writer.close()
