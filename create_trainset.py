import cv2
import numpy as np
import pickle as pk
import os
from image_preprocessing import crop_image

# This code creates a training set out of the video data

videos = os.listdir("demo_videos")

vid_dir = "demo_videos/"

all_video_data = []
i = 0
for vid in videos:
    test_vid = vid_dir+vid
    cap = cv2.VideoCapture(test_vid)
    vid_array = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = crop_image(frame)
            img = cv2.resize(frame, (64, 36))
            vid_array.append(img)
    except cv2.error:
        pass
    cap.release()
    vid_array = np.array(vid_array)
    if vid_array.shape[0]>500:
        print(vid_array.shape, "saved")
        # np.save( "numpy_vids_36_64/vid%d"%i, vid_array)
        all_video_data.append(vid_array)
        i=i+1
    else:
        print(vid_array.shape, "skipping...")


all_video_data = np.array(all_video_data)
np.save("video_data.npy", all_video_data)

print(all_video_data.shape)

