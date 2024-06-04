import sys
import rclpy
from rclpy.node import Node
from yolo_strategy_interfaces.srv import YoloStrategy

import cv2
import numpy as np
import pyrealsense2 as rs

def take_pics():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    a = 0
    # Start streaming
    pipeline.start(config)

    # Instructions for user
    print('Press m to take pictures')
    print('Press q to quit camera\n')
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # if not color_frame:
        #     continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Resize color image 
        # resized_color_image = cv2.resize(table_outline, dsize=(1280,720), interpolation=cv2.INTER_AREA)    

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key=cv2.waitKey(1)
        if key&0xFF==ord('m'):
            a=a+1
            print("picture taken")
            # cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_ball.jpg',color_image)
            cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/pics/PoolBall_2201.jpg',color_image)
        if key&0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
    # Stop streaming
    pipeline.stop()


class TestYoloService(Node):

    def __init__(self):
        super().__init__('test_yolo_service')
        self.test_client = self.create_client(YoloStrategy, 'yolo_strategy')
        while not self.test_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = YoloStrategy.Request()

    def send_request(self, a):
        self.req.send_position = a
        self.future = self.test_client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main():
    rclpy.init()

    test_service = TestYoloService()
    response = test_service.send_request(int(sys.argv[1]))
    test_service.get_logger().info(
        'Response:{}\n'.format(response.current_position))
    test = response.current_position
    print('test',test)

    test_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()