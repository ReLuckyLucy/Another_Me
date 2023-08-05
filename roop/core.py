#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
if not 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    del torch
import tensorflow

import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args(newface,oldface,savepath) -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', default=newface)
    program.add_argument('-t', '--target', help='select an target image or video', default=oldface)
    program.add_argument('-o', '--output', help='select output file or directory',default=savepath)
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=100, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)  # type: ignore
    roop.globals.headless = roop.globals.source_path and roop.globals.target_path and roop.globals.output_path
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)#返回可用的执行提供者
    roop.globals.execution_threads = args.execution_threads#返回可用的执行提供者

    print
def encode_execution_providers(execution_providers: List[str]) -> List[str]:#编码执行提供者
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]#返回可用的执行提供者


def decode_execution_providers(execution_providers: List[str]) -> List[str]:#解码执行提供者
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))#返回可用的执行提供者
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]#返回可用的执行提供者


def suggest_execution_providers() -> List[str]:#建议执行提供者
    return encode_execution_providers(onnxruntime.get_available_providers())#返回可用的执行提供者


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:#限制资源
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')#获取GPU列表
    for gpu in gpus:#遍历GPU列表
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [#设置GPU的内存使用上限
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)#设置GPU的内存使用上限
        ])
    # limit memory usage
    if roop.globals.max_memory:#如果设置了最大内存
        memory = roop.globals.max_memory * 1024 ** 3#将最大内存转换为字节
        if platform.system().lower() == 'darwin':#如果是macOS
            memory = roop.globals.max_memory * 1024 ** 6#将最大内存转换为字节
        if platform.system().lower() == 'windows':#如果是windows
            import ctypes#加载ctypes库
            kernel32 = ctypes.windll.kernel32#加载kernel32.dll
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))#设置进程工作集大小
        else:
            import resource#加载resource库
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))#设置进程数据段大小


def pre_check() -> bool:#预检查
    if sys.version_info < (3, 9):#如果python版本小于3.9
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')#更新状态
        return False
    if not shutil.which('ffmpeg'):#如果没有安装ffmpeg
        update_status('ffmpeg is not installed.')#更新状态
        return False
    return True#


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:#更新状态
    print(f'[{scope}] {message}')#打印状态
    if not roop.globals.headless:#如果不是无头模式
        ui.update_status(message)#更新状态
#

def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):#遍历帧处理器
        if not frame_processor.pre_start():#如果帧处理器没有预启动
            return
    # process image to image
    if has_image_extension(roop.globals.target_path):#如果目标文件是图片
        if predict_image(roop.globals.target_path):#如果目标文件是图片
            destroy()#销毁
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)#复制图片
        # process frame
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):#遍历帧处理器
            update_status('Progressing...', frame_processor.NAME)#更新状态
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)#处理图片
            frame_processor.post_process()#后处理
        # validate image
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')#更新状态
        else:
            update_status('Processing to image failed!')#更新状态
        return
    # process image to videos
    if predict_video(roop.globals.target_path):#如果目标文件是视频
        destroy()
    update_status('Creating temporary resources...')#更新状态
    create_temp(roop.globals.target_path)#创建临时文件夹
    # extract frames
    if roop.globals.keep_fps:#如果保持fps
        fps = detect_fps(roop.globals.target_path)#检测fps
        update_status(f'Extracting frames with {fps} FPS...')#更新状态
        extract_frames(roop.globals.target_path, fps)#提取帧
    else:
        update_status('Extracting frames with 30 FPS...')#更新状态  
        extract_frames(roop.globals.target_path)#提取帧
    # process frame
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)#获取临时帧路径
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):#遍历帧处理器
            update_status('Progressing...', frame_processor.NAME)#更新状态
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)#处理视频
            frame_processor.post_process()#后处理
    else:
        update_status('Frames not found...')#更新状态
        return
    # create video
    if roop.globals.keep_fps:#如果保持fps
        fps = detect_fps(roop.globals.target_path)#检测fps
        update_status(f'Creating video with {fps} FPS...')#更新状态
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')#更新状态
        create_video(roop.globals.target_path)
    # handle audio
    if roop.globals.skip_audio:#如果跳过音频
        move_temp(roop.globals.target_path, roop.globals.output_path)#移动临时文件
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:#  如果保持fps
            update_status('Restoring audio...')#更新状态
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')#更新状态
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    # clean temp
    update_status('Cleaning temporary resources...')#更新状态
    clean_temp(roop.globals.target_path)#清理临时文件
    # validate video
    if is_video(roop.globals.target_path):#如果目标文件是视频
        update_status('Processing to video succeed!')#更新状态
    else:
        update_status('Processing to video failed!')#   更新状态


def destroy() -> None:#销毁
    if roop.globals.target_path:#如果目标文件存在
        clean_temp(roop.globals.target_path)#清理临时文件
    sys.exit()#退出



def run(newface,oldface,savepath) -> None:#运行
    parse_args(newface,oldface,savepath)#解析参数
    if not pre_check():
        return#预检查
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):#遍历帧处理器
        if not frame_processor.pre_check():#如果帧处理器没有预检查
            return
    limit_resources()#限制资源
    if roop.globals.headless:   # 如果是无头模式
        start() # 开始
    else:
        window = ui.init(start, destroy)    # 初始化窗口
        window.mainloop()   # 进入消息循环
 