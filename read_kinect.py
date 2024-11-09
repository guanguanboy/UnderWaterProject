from pykinect2 import PyKinectRuntime, PyKinectV2
import cv2
import numpy as np

# 初始化 Kinect 运行时，只启用颜色和深度流
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

# 读取帧并保存图像的函数
def save_frames():
    frame_count = 0
    while True:
        # 检查是否有新的颜色帧
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_image = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)  # 将1D数组转换为3D（1080x1920x4）
            cv2.imwrite(f"color_frame_{frame_count}.png", color_image)  # 保存RGB图像
        
        # 检查是否有新的深度帧
        if kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
            depth_image = depth_frame.reshape((424, 512)).astype(np.uint16)  # 将1D数组转换为2D（424x512）
            cv2.imwrite(f"depth_frame_{frame_count}.png", depth_image)  # 保存深度图像

        frame_count += 1
        if frame_count > 100:  # 停止条件
            break

# 执行保存函数
save_frames()

# 释放 Kinect 资源
kinect.close()
