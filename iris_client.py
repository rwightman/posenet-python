import cv2
import rospy
from sensor_msgs.msg import Image
import socket
import sys
import numpy as np

def callback(data):
    try:
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("192.168.123.108", 33333))
        client.sendall(bytearray(img))
        client.close()

    except Exception as err:
        print(err)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/camera1/image_raw', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
