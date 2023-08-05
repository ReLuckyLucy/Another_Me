import threading
from typing import Any, Optional, List
import insightface
import numpy

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()#创建锁


def get_face_analyser() -> Any:#获取人脸分析器
    global FACE_ANALYSER#全局变量FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:#如果FACE_ANALYSER为None
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)#创建人脸分析器
            FACE_ANALYSER.prepare(ctx_id=0)#准备人脸分析器
    return FACE_ANALYSER#返回人脸分析器


def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:#获取一张人脸
    many_faces = get_many_faces(frame)#获取多张人脸
    if many_faces:
        try:
            return many_faces[position]#返回第position张人脸
        except IndexError:#如果发生索引错误
            return many_faces[-1]#返回最后一张人脸
    return None


def get_many_faces(frame: Frame) -> Optional[List[Face]]:#获取多张人脸
    try:
        return get_face_analyser().get(frame)#获取人脸分析器
    except ValueError:#如果发生值错误
        return None#返回None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:#寻找相似人脸
    many_faces = get_many_faces(frame)#获取多张人脸
    if many_faces:#如果有人脸
        for face in many_faces:#遍历人脸
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):#如果人脸有normed_embedding属性和参考人脸有normed_embedding属性
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))#计算距离
                if distance < roop.globals.similar_face_distance:#如果距离小于相似人脸距离
                    return face
    return None
