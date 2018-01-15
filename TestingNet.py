import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

test = pickle.load(open("time_steps/e6.p", "rb"))

plt.plot(range(len(test)), test)
plt.ylim(0,500)
plt.show()

# test2 = np.load("numpy_vids/vid82.npy")
#
# for img in test2:
#     cv2.imshow("Image",img)
#     cv2.waitKey(10)