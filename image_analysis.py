#!pip install tensorflow tensorflow-gpu opencv-python matplotlib

import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

!pip install tensorflow-gpu

!pip list

import tensorflow as tf
import os

os.path.join('data','happy')

os.listdir('data')

gpus = tf.config.experimental.list_physical_devices('GPU')

gpus

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

gpus = tf.config.experimental.list_physical_devices('CPU')

gpus

import cv2
import imghdr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_dir= 'data'

os.listdir(data_dir)

os.listdir(os.path.join(data_dir,'happy'))

image_exts= ['jpeg','jpg','bmp','png']

image_exts

image_exts[0]

image_exts[2]

image_exts[1]

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path= os.path.join(data_dir,image_class,image)
        try:
            img= cv2.imread(image_path)
            tip= imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            #os.remove(image_path)

for image_class in os.listdir(data_dir):
    print(image_class)

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        print(image)

img=cv2.imread(os.path.join('data','happy', 'happiness.jpg'))

img

type(img)

img.shape

plt.imshow(img)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path= os.path.join(data_dir,image_class,image)
        try:
            img= cv2.imread(image_path)
            tip= imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            #os.remove(image_path)



"""# Load Data"""

import numpy as np
from matplotlib import pyplot as plt

tf.keras.utils.image_dataset_from_directory??

data = tf.keras.utils.image_dataset_from_directory('data')

data

data_iterator= data.as_numpy_iterator()

data_iterator

batch= data_iterator.next()

batch

len(batch)

#Get another batch from the iterator
batch = data_iterator.next()

batch[0].shape

#Class 1 = sad people
#class 2 = happy people

batch[1]

fig,ax = plt.subplots(ncols=4,figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

batch[1]

scaled= batch[0]/255

scaled.min()

scaled.max()

"""# Preprocce Data

## Scale data
"""

data = data.map(lambda x,y: (x/255,y))

data.as_numpy_iterator().next()

data.as_numpy_iterator().next()[0].max()

scaled_iterator= data.as_numpy_iterator()



batch= scaled_iterator.next()

batch[0].max()

batch[0].min()

fig,ax = plt.subplots(ncols=4,figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])



"""## Split data"""

len(data)

train_size= int(len(data)*.7)
val_size= int(len(data)*.2)
test_size= int(len(data)*.1)

val_size

test_size

train_size

train_size+test_size+val_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

len(test)

len(train)

len(val)

"""## Deep Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())


model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss= tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

model.summary()



"""## Train

"""

logdir = 'logs'

tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=logdir)



hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])



"""## Plot Performance"""

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()



"""## Evaluate"""

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

print(pre.result(), re.result(), acc.result())



"""## Test"""

import cv2

img = cv2.imread('/content/data/happy/1902587.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

yhat

if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')



"""## Save the model

"""

from tensorflow.keras.models import load_model

import os
model.save(os.path.join('models','happy_sad_model.keras'))

import os
from tensorflow.keras.models import load_model

# Load the model by specifying the correct path
new_model = load_model(os.path.join('models', 'happy_sad_model.keras'))

os.path.join('models', 'happy_sad_model.keras')

new_model.predict(np.expand_dims(resize/255, 0))

yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')


