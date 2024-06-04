"""
    YOLO & Python 環境需求
        1. .data
        2. .name
        3. .cfg
        4. .weight
        5. darknet(shared library)
        6. darknet.py
        7. libdarknet.so
"""

import pyrealsense2 as rs           # 版本目前不支援python3.10
import time
import cv2
import numpy as np
import darknet



# video = cv2.VideoCapture("/home/weng/ICLAB/output2.avi")

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out_video = cv2.VideoWriter('demo_YOLO_2.avi', fourcc, 15.0, (1280,  720))

# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# pipeline.start(config)
# sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
# sensor.set_option(rs.option.auto_exposure_priority, True)


"""
神經網路檔案位置_檢測全部拼圖
"""
ALL_cfg_path = './cfg/yolov4-obj.cfg'
ALL_weights_path = './cfg/weights/ALL/yolov4-obj_best.weights'
ALL_data_path = './cfg/hiwin_C_WDA_v4.data'


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
        print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), xmin, ymin, xmax, ymax))
        
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


# def detect_Angle(img,thresh=0.3):
#     out,detections = image_detection(img,Angle_network, Angle_class_names, Angle_class_colors,thresh)
#     out2,puzzle_imformation = draw_boxes(detections, img, Angle_class_colors)

#     cv2.imshow('out2', out2)
#     cv2.waitKey(1000)
#     # cv2.destroyAllWindows()

#     return puzzle_imformation



"""
主程式
    程式流程:
    1. 檢測影像
    2. 在原圖繪製結果
    3. 輸出影像
"""
if __name__ == "__main__":

    img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/detect_second_ball.jpg', cv2.IMREAD_UNCHANGED)
    # cv2.imshow('org img', img)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    print('Original Dimensions : ',img.shape)

    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dsize=dim, interpolation = cv2.INTER_AREA)

    print('Resized Dimensions : ',resized.shape)

    # cv2.imshow("Resized image", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #while(1):
        # frames = pipeline.wait_for_frames()
        # image = frames.get_color_frame()
        # image = np.asanyarray(image.get_data())

    out2, ballinfo = detect_ALL(img)  
    print(ballinfo)  
    cv2.imwrite("detected_upper.jpg",out2)
    cv2.imshow('detected img',out2)
    cv2.waitKey()

    # print("=============================================")
    # image = cv2.imread(img_name)
    # # detect_Angle(image) 
    # cv2.waitKey(3000)

    # out,detections = image_detection(image,ALL_network, ALL_class_names, ALL_class_colors,0.8)

    # out2, puzzle_imformation = draw_boxes(detections, image, ALL_class_colors)
    # # out_video.write(out2)

    # #cv2.imshow('out', out)      # YOLO壓縮大小
    # cv2.imshow('out2', out2)    # 原圖
    # cv2.waitKey(3000)




    # while 1:
    #     # ret, image = video.read()
    #     detections = 0  #detections歸零(以後迴圈可能會有影響)
    #     out,detections = image_detection(image,ALL_network, ALL_class_names, ALL_class_colors,0.8)

    #     out2, puzzle_imformation = draw_boxes(detections, image, ALL_class_colors)
    #     # out_video.write(out2)

    #     #cv2.imshow('out', out)      # YOLO壓縮大小
    #     cv2.imshow('out2', out2)    # 原圖
    #     # cv2.waitKey(0)


    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

# video.release()
# out_video.release()
# cv2.destroyAllWindows()







# 參考資料
# https://blog.csdn.net/Cwenge/article/details/80389988
# https://www.google.com/search?q=libdarknet.so+%E6%89%BE%E4%B8%8D%E5%88%B0&oq=libdarknet.so+%E6%89%BE%E4%B8%8D%E5%88%B0&aqs=chrome..69i57j33i160l2.9119j0j4&sourceid=chrome&ie=UTF-8