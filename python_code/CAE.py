import os
import pickle as pk
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Reshape
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import transform
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



with open("datos_aumentados.pkl","rb") as f:
    datos = pk.load(f)
y = np.loadtxt('indices.txt',delimiter = ',')

x_train, x_test, y_train, y_test = train_test_split(datos,y,test_size=0.1, random_state=69)

autoencoder = Sequential()
encoder = Sequential()
decoder = Sequential()


# Encoder Layers
encoder.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=(64,64,1)))
encoder.add(Conv2D(32, (3, 3), activation='relu', padding='valid',strides=2))
encoder.add(Conv2D(64, (3, 3), activation='relu', padding='valid',strides=2))
encoder.add(Conv2D(32, (3, 3), activation='relu', padding='valid',strides=2))
encoder.add(Conv2D(16, (3, 3), activation='relu', padding='valid',strides=2))
encoder.add(Conv2D(10, (3, 3), activation='relu', padding='same',strides=2))


encoder.summary()


# Decoder Layers
decoder.add(Conv2D(10, (3, 3), activation='relu', padding='same',input_shape=(1,1,10)))
#decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
#decoder.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
decoder.add(UpSampling2D((2, 2)))
#decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
#decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(1, (3, 3), padding='same'))


decoder.summary()


autoencoder.add(encoder)

autoencoder.add(Flatten())

autoencoder.add(Reshape((1, 1, 10)))

autoencoder.add(decoder)

encoder.summary()

autoencoder.summary()

def resize_batch(imgs):
    imgs = imgs.reshape((-1, 63, 63, 1))
    resized_imgs = np.zeros((imgs.shape[0], 64, 64, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (64, 64))
    return resized_imgs



nuevo = resize_batch(x_train)
print(np.shape(nuevo))
test = resize_batch(x_test)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
history = autoencoder.fit(nuevo, nuevo, epochs=80,   batch_size=100, validation_data=(test, test))



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('entrenamiento.png')


model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")

model_json = encoder.to_json()
with open("encoder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder.save_weights("encoder.h5")
print("Saved model to disk")


model_json = decoder.to_json()
with open("decoder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
decoder.save_weights("decoder.h5")
print("Saved model to disk")


