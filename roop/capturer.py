from typing import Optional
import cv2

from roop.typing import Frame


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:#获取视频帧
    capture = cv2.VideoCapture(video_path)#打开视频
    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)#获取视频帧数    
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))#设置视频帧位置
    has_frame, frame = capture.read()#读取视频帧
    capture.release()#释放视频
    if has_frame:#如果有视频帧
        return frame#返回视频帧
    return None#否则返回None


def get_video_frame_total(video_path: str) -> int:#获取视频帧数
    capture = cv2.VideoCapture(video_path)#打开视频
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))#获取视频帧数
    capture.release()#释放视频
    return video_frame_total#返回视频帧数
