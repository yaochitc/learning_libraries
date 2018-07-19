import numpy as np
from skimage.color import gray2rgb, rgb2gray
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Input
from sklearn.datasets import fetch_olivetti_faces
from lime.lime_image import LimeImageExplainer

faces = fetch_olivetti_faces()

# make each image color so lime_image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in faces.data.reshape((-1, 64, 64))],0)
y_vec = faces.target.astype(np.uint8)

model = Sequential()
model.add(Input())
model.add(Conv2D())
model.add(Dense())

model.compile('adam', loss='cross_entropy')

