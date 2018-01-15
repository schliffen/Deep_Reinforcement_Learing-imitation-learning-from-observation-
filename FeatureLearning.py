from sklearn.decomposition import PCA
import numpy as np
from image_preprocessing import crop_image_gray

expert_vid = np.load("numpy_vids/vid%d.npy" % 81)

expert_vid = list(map(lambda x: crop_image_gray(x).ravel(), expert_vid))

pca = PCA(n_components=20)

Xt=pca.fit_transform(expert_vid)

print(Xt.size)

