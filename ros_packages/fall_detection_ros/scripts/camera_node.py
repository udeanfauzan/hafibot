#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.header.stamp = rospy.Time.now()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg

def main():
    rospy.init_node('camera_node', anonymous=True)
    pub = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=1)
    
    device_id = rospy.get_param('~video_device', '/dev/video0')
    width = rospy.get_param('~image_width', 640)
    height = rospy.get_param('~image_height', 480)
    
    # Extract integer ID from device_id string (e.g. '/dev/video0' -> 0)
    try:
        cam_id = int(device_id.replace('/dev/video', ''))
    except ValueError:
        cam_id = 0
        
    # Menggunakan GStreamer Pipeline khusus NVIDIA Jetson (nvarguscamerasrc)
    # - Ambil video 1080p@30fps dari sensor IMX219 (mode native paling stabil)
    # - 'queue leaky=downstream': katup pengaman, buang frame lama jika GPU sibuk
    # - 'nvvidconv': resize ke 640x480 via hardware GPU (tanpa CPU)
    # - 'appsink max-buffers=1 drop=true sync=false': selalu ambil frame terbaru saja
    gstreamer_pipeline = (
        "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
        "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "nvvidconv ! "
        "video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true sync=false"
    ).format(w=width, h=height)
    
    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        rospy.logerr(f"Gagal membuka kamera: {device_id}")
        return

    rospy.loginfo(f"CSI Camera berjalan (1080p -> {width}x{height})")
    
    # Loop rate 15 FPS — memberi GPU ruang bernapas untuk MediaPipe
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            msg = cv2_to_imgmsg(frame)
            pub.publish(msg)
        else:
            rospy.logwarn_throttle(5, "Gagal membaca frame kamera.")
        rate.sleep()
        
    cap.release()

if __name__ == '__main__':
    main()
