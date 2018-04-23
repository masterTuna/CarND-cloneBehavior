import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import os
import random
from pandas import DataFrame, Series
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
import sklearn

IMG_PATH = r'../data'
straight_drive_drop_rate = 50 # drop percentage of data with angle 0 
#IMG_PATH = r'../example_data'
lines = []
with open(IMG_PATH + r'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # ignore the title line
        for line in reader:
                # run filter for angle=0 data
                if float(line[3]) == 0:
                    if random.randint(1, 100) > straight_drive_drop_rate:
                        continue
                lines.append(line)
train_samples, valid_samples = train_test_split(lines, test_size = 0.2)

def getImage(read_path):
    filename = read_path.split('/')[-1]
    cur_path = IMG_PATH + '/IMG/' + filename
    img_org = cv2.imread(cur_path)
    image = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    return image

def generator(samples, batch_size=128):
    flip_rate = 75 # percent. define the percent of images with reasonable angle to be flipped
    n =len(samples)
    d_angle = .25
    adjust_angle = {'left':d_angle, 'center':0, 'right':-d_angle}
    camera = ['center', 'left', 'right']

    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, n, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                select_camera = random.randint(0, 2)
                image = getImage(batch_sample[select_camera])
                measurement = float(batch_sample[3]) + adjust_angle[camera[select_camera]]
                if random.randint(1, 100) < flip_rate:
                    image = np.fliplr(image)
                    measurement = -measurement
                images.append(image)
                measurements.append(measurement)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
                
def show_data_dist(lines):
    angles = []
    for line in lines:
        angles.append(line[3])
    se = Series(angles)
    print("Total data: ", len(se))
    plt.figure(0)
    se.hist(bins=50)
    plt.title("data distribution")
    plt.savefig("./images_doc/dist.png")

# ------------------------------------------------------------------------------
# processing data
# ------------------------------------------------------------------------------
#
show_data_dist(lines)

train_gen = generator(train_samples, 32)
validation_gen = generator(valid_samples, 32)
#
print("Build Model")
keep_prob = 0.5
# keras model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))) # image processing layers: crop, normalization
model.add(Cropping2D(cropping=((70, 24), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64,3,3, activation='relu'))
#model.add(Dropout(keep_prob))
model.add(Convolution2D(64,3,3, activation='relu'))
#model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(100))
#model.add(Activation('relu'))
model.add(Dropout(keep_prob))
model.add(Dense(50))
#model.add(Activation('relu'))
model.add(Dropout(keep_prob))
model.add(Dense(10))
#model.add(Activation('relu'))
model.add(Dense(1)) # angle
#plot_model(model, to_file='./images_doc/model.png', show_shapes=True)
model.compile(loss='mse', optimizer='adam')

# history
history_obj = model.fit_generator(train_gen, 
                                  samples_per_epoch=len(train_samples*2), 
                                  validation_data=validation_gen,
                                  nb_val_samples=len(valid_samples),
                                  nb_epoch=4
                                  )
print(history_obj.history.keys())
plt.figure(1)
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model MSError loss')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'])
plt.savefig('./images_doc/train_history.png')

# save network
model.save('model.h5')
exit()
