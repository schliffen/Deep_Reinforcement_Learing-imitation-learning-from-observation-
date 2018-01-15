import cv2
import numpy as np
import matplotlib.pyplot as plt
# we are going to apply some pre-processing steps on the image


def crop_image_gray(image):
    return image[166:336, 100:500]


def normalize_img(image_vec):
    return (image_vec-np.mean(image_vec))/np.std(image_vec)


def crop_image(image):
    # image should be 3 channels
    try:
        im_ch1 = image.T[0].T[166:336, 100:500]/255.
        im_ch2 = image.T[1].T[166:336, 100:500]/255.
        im_ch3 = image.T[2].T[166:336, 100:500]/255.
        out_image = convert_to_3_channel_image(im_ch1, im_ch2, im_ch3)
    except AttributeError:
        print("end of sequence...")
    else:
        return out_image


def convert_to_3_channel_image(ch1, ch2, ch3):
    return np.array([ch1.T, ch2.T, ch3.T]).T


# #
# im = cv2.imread("temp/test1.PNG")
# im2 = crop_image(im)
# im2 = cv2.resize(im2,(64, 36))
# print(im2.shape)
# im2 = im2.T[0].T
# # print(im2)
# print(im2.shape)
# plt.imshow(im2, cmap="gray")
# plt.show()
