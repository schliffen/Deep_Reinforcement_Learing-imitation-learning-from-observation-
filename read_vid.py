import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_preprocessing import crop_image

# read video data
vid_data = np.load("numpy_vids_36_64/vid3.npy")
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 500, 500)

print(vid_data.shape)
# for image in vid_data:
#     cv2.imshow("image", image)
#     cv2.waitKey(100)

# read image from video
# img1 = cv2.imread("temp/test1.png", 0)
# img2 = cv2.imread("temp/test2.png", 0)
# print(img1.shape)
# img1 = cv2.resize(img1, (64, 36))
# full_img = np.array([img1, img1])
#
# cv2.imshow("image", img1)
# cv2.waitKey(5000)


# plt.imshow(img1, cmap="gray")
# plt.show()
#
# print(img1.shape)
# print(full_img.shape)

# vid_data = np.load("video_data.npy")

# print(vid_data)

