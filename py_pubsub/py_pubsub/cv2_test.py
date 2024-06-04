import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./pics/PoolBall_2201.jpg')
detected_img = cv2.imread('/home/zack/work/ROS2_ws/detected_img.jpg')
print('Original Dimensions : ',img.shape)
scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dsize=dim, interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized.shape)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

# img = plt.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/pics/PoolBall_2201.jpg')
# plt.imshow(img)
# plt.show()
