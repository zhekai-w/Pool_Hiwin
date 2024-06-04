import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray


import cv2
import py_pubsub.darknet as darknet
import numpy as np

"""
神經網路檔案位置_檢測全部拼圖
"""
ALL_cfg_path ="/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/yolov4-obj.cfg"      #'./cfg/yolov4-obj.cfg'
ALL_weights_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/weights/ALL/yolov4-obj_best.weights'
ALL_data_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/hiwin_C_WDA_v4.data'


"""
載入神經網路
"""
ALL_network, ALL_class_names, ALL_class_colors = darknet.load_network(
        ALL_cfg_path,
        ALL_data_path,
        ALL_weights_path,
        batch_size=1
)



"""
影像檢測
    輸入:(影像位置,神經網路,物件名稱集,信心值閥值(0.0~1.0))
    輸出:(檢測後影像,檢測結果)
    註記:
"""
def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections



"""
座標轉換
    輸入:(YOLO座標,原圖寬度,原圖高度)
    輸出:(框的左上座標,框的右下座標)
    註記:
"""
def bbox2points(bbox,W,H):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """ 
    width = darknet.network_width(ALL_network)      # YOLO壓縮圖片大小(寬)
    height = darknet.network_height(ALL_network)    # YOLO壓縮圖片大小(高)

    x, y, w, h = bbox                           # (座標中心x,座標中心y,寬度比值,高度比值)
    x = x*W/width
    y = y*H/height
    w = w*W/width
    h = h*H/height
    # 輸出框座標_YOLO格式
    # print("     (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(x, y, w, h))
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    
    return xmin, ymin, xmax, ymax



"""
原圖繪製檢測框線
    輸入:(檢測結果,原圖位置,框線顏色集)
    輸出:(影像結果)
    註記:
"""
def draw_boxes(detections, image, colors):
    ball_imformation = [[-999 for i in range(4)] for j in range(20)]
    i = 0

    H,W,_ = image.shape                      # 獲得原圖長寬

    # cv2.line(image,(640,0),(640,720),(0,0,255),5)

    for label, confidence, bbox in detections:
        xmin, ymin, xmax, ymax = bbox2points(bbox,W,H)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
        # 輸出框座標_加工格式座標(左上點座標,右上點座標)
        #print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), xmin, ymin, xmax, ymax))
        
        mx = float(xmax + xmin)/2
        my = float(ymax + ymin)/2

        # cv2.circle(image, (int(mx),int(my)), 33, (0,0,255), 3)
        if label == 'C':
            ball_imformation[i] = [0.0, float(confidence), mx, my]
        elif label == 'M':
            ball_imformation[i] = [1.0, float(confidence), mx, my]
        i+=1
        

    return image, ball_imformation


def detect_ALL(img,thresh=0.8):
    out,detections = image_detection(img,ALL_network, ALL_class_names, ALL_class_colors,thresh)
    out2, ball_imformation= draw_boxes(detections, img, ALL_class_colors)

    # cv2.imshow('out2', out2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return out2, ball_imformation


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.flag_yolo_callback)
        # self.flag_yolo_callback()
        self.flag = 0
        self.i = 0.0
        self.j = 0
        self.a=[]
        self.n = 2201

    def timer_callback(self):
        msg = Float32MultiArray()
        self.a.append(self.i)
        msg.data = self.a
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%f"' % msg.data[self.j])
        self.i += 1.0
        self.j += 1

    def flag_yolo_callback(self):
        while(input('enter p to publish:\n') == 'p'):
            ballmsg = Float32MultiArray()
            
            img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/pics/PoolBall_%d.jpg'%(self.n))
        
            print('Original Dimensions : ',img.shape)

            # scale_percent = 80 # percent of original size
            # width = int(img.shape[1] * scale_percent / 100)
            # height = int(img.shape[0] * scale_percent / 100)
            # dim = (width, height)
            
            # # resize image
            # resized = cv2.resize(img, dsize=dim, interpolation = cv2.INTER_AREA)

            # print('Resized Dimensions : ',resized.shape)

            out2, ballinfo = detect_ALL(img)  
            # check for non float value in ballinfo since ballmsg.data does not accept data other than float
            cnt = 0
            for i in range(len(ballinfo)):
                if ballinfo[i][1] != -999:
                    cnt += 1
                else:
                    break
            #flatten list of 2darray to 1darray
            flat_list = []
            for i in range(0,cnt):
                flat_list.extend(ballinfo[i])

            cv2.imwrite("detected_img.jpg",out2)
            
            
            ballmsg.data = flat_list
            self.publisher_.publish(ballmsg)
            self.get_logger().info('Publishing ball location:{}\n'.format(ballmsg.data))
            self.n += 1
            
            if self.n > 2210:
                self.flag = 1
            else:
                pass
            # REMEMBER TO CLEAR THIS        
            # self.flag = 1
        else:
            pass

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()