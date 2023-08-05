import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm#导入tqdm模块

import roop.globals#导入roop.globals模块

TEMP_DIRECTORY = 'temp'#
TEMP_VIDEO_FILE = 'temp.mp4'#

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':#
    ssl._create_default_https_context = ssl._create_unverified_context#


def run_ffmpeg(args: List[str]) -> bool:#运行ffmpeg
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]#ffmpeg命令
    commands.extend(args)#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)#subprocess.check_output()函数用于执行外部命令。它返回命令执行后的输出。
        return True
    except Exception:
        pass
    return False


def detect_fps(target_path: str) -> float:#检测fps
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]#ffprobe命令
    output = subprocess.check_output(command).decode().strip().split('/')#subprocess.check_output()函数用于执行外部命令。它返回命令执行后的输出。
    try:
        numerator, denominator = map(int, output)#map() 会根据提供的函数对指定序列做映射。
        return numerator / denominator#返回fps
    except Exception:
        pass
    return 30


def extract_frames(target_path: str, fps: float = 30) -> bool:#提取帧
    temp_directory_path = get_temp_directory_path(target_path)#获取临时目录路径
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100#获取临时帧质量
    return run_ffmpeg(['-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', 'fps=' + str(fps), os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)])#运行ffmpeg


def create_video(target_path: str, fps: float = 30) -> bool:#创建视频
    temp_output_path = get_temp_output_path(target_path)#获取临时输出路径
    temp_directory_path = get_temp_directory_path(target_path)#获取临时目录路径
    output_video_quality = (roop.globals.output_video_quality + 1) * 51 // 100#获取输出视频质量
    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format), '-c:v', roop.globals.output_video_encoder]#ffmpeg命令
    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:#如果输出视频编码器在['libx264', 'libx265', 'libvpx']中
        commands.extend(['-crf', str(output_video_quality)])#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:#如果输出视频编码器在['h264_nvenc', 'hevc_nvenc']中
        commands.extend(['-cq', str(output_video_quality)])#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
    return run_ffmpeg(commands)


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(['-hwaccel', 'auto', '-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.' + roop.globals.temp_frame_format)))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    if source_path and target_path and output_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:#判断是否为图片扩展名
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))#如果image_path以('png', 'jpg', 'jpeg', 'webp')中的一种结尾，返回True#否则返回False


def is_image(image_path: str) -> bool:#判断是否为图片
    if image_path and os.path.isfile(image_path):#如果image_path存在且为文件
        mimetype, _ = mimetypes.guess_type(image_path)#mimetypes.guess_type()函数用于根据文件扩展名猜测文件的MIME类型#mimetype为文件的MIME类型，_为文件的编码方式
        return bool(mimetype and mimetype.startswith('image/'))#如果mimetype存在且以'image/'开头，返回True#否则返回False
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
