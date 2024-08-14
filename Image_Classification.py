import cv2
import imghdr
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

image_exts = ['jpeg', 'jpg', 'bmp', 'png']
data_dir = "C:/Users/ACER/Downloads/New folder/images"

def preprocess_images(data_dir):
    for image_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, image_class)
        for image in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image)
            try:
                
                img = cv2.imread(image_path)
                if img is None:
                    print(f'Image file is not readable: {image_path}')
                    os.remove(image_path)
                    continue
                
               
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f'Image not in ext list: {image_path}')
                    os.remove(image_path)
                    
            except Exception as e:
                print(f'Issue with image {image_path} - {e}')
                if os.path.exists(image_path):
                    os.remove(image_path)


preprocess_images(data_dir)


try:
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
except Exception as e:
    print(f'Error loading dataset: {e}')
    exit()

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

data = data.map(lambda x, y: (x / 255, y))


train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


hist = model.fit(train, epochs=20, validation_data=val)


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.title('Loss')
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
plt.title('Accuracy')
plt.legend(loc="upper left")
plt.show()


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}')
print(f'Recall: {re.result().numpy()}')
print(f'Accuracy: {acc.result().numpy()}')


img = cv2.imread("C:/Users/ACER/Downloads/test.jpg")
if img is None:
    print('Invalid image file')
else:
    plt.imshow(img)
    plt.show()
    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    if yhat > 0.5:
        print('Predicted class is Dog')
    else:
        print('Predicted class is Cat')


from tensorflow.keras.models import save_model

model.save(os.path.join('models', 'imageclassifier.h5'))
