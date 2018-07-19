import numpy as np
from keras import Sequential
from keras.layers import Dense

data = np.random.random((1000, 32))
label = np.random.random((1000, 10))

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(32, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy')

model.fit(data, label, epochs=100)

model.save('my_model.h5')

