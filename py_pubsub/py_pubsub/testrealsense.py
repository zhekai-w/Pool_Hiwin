import pyrealsense2 as rs

width = 1920
height = 1080
fps = 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
# config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

# ストリーミング開始

profile = pipeline.start(config)
# depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
color_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

# print("depth_intrinsics")
# print(depth_intrinsics)
print("color_intrinsics")
print(type(color_intrinsics))
print(color_intrinsics)

# 歪みを補正（変換）
x, y = 1300,1200  # 変換したい座標
depth = 1

pixel = [x, y]
# point = rs.rs2_deproject_pixel_to_point(color_intrinsics, pixel, depth)
# print('org point:',point)

calibrated_intrinsics_f = [1362.38, 1360.45]
calibrated_intrinsics_pp = [938.315, 552.935]

dis_coeffs = [0.0693826933, 0.445315521, 0.00291064076, -0.000845071017, -1.99098719]

_intrinsics = rs.intrinsics()
_intrinsics.width = width
_intrinsics.height = height
_intrinsics.ppx = calibrated_intrinsics_pp[0]
_intrinsics.ppy = calibrated_intrinsics_pp[1]
_intrinsics.fx = calibrated_intrinsics_f[0]
_intrinsics.fy = calibrated_intrinsics_f[1]
# _intrinsics.model = cameraInfo.distortion_model
_intrinsics.model  = rs.distortion.none
_intrinsics.coeffs = dis_coeffs
ca_point = rs.rs2_deproject_pixel_to_point(_intrinsics, pixel, depth)
print('calibrated point:',ca_point)

# # カメラ座標をスクリーン座標に変換（歪みなし）
# x_ = int(point[0] * color_intrinsics.fx + color_intrinsics.ppx)
# y_ = int(point[1] * color_intrinsics.fy + color_intrinsics.ppy)
# print('default intrinsics x:',x_)
# print('default intrinsics y:',y_)


x__ = int(ca_point[0] * calibrated_intrinsics_f[0] + calibrated_intrinsics_pp[0])
y__ = int(ca_point[1] * calibrated_intrinsics_f[1] + calibrated_intrinsics_pp[1])
print('calibrated intrinsics x:',x__)
print('calibrated intrinsics y:',y__)

# pipeline.stop()


# calibrated_intrinsics
# [[1.36238016e+03 0.00000000e+00 9.38315245e+02]
#  [0.00000000e+00 1.36045249e+03 5.52935948e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# [[ 6.93826933e-02  4.45315521e-01  2.91064076e-03 -8.45071017e-04 -1.99098719e+00]]
