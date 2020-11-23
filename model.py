# processing
import cv2
import numpy as np
import math
import sklearn

# network architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# data processiong
from keras.layers import Lambda, Cropping2D
from sklearn.model_selection import train_test_split

# filesystem
import os
import csv

#read input file names / angles form CSV and store as lines
def read_data_list(data_directory="./data/"):
    lines = []
    with open(data_directory+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers #skip header
        for line in reader:
            lines.append(line)
    return lines


def generator(samples, data_directory="./data/", batch_size=32, augment_flip = False, add_side_images = False):
    num_samples = len(samples)
    print("num_samples:",num_samples)
    if add_side_images == True & augment_flip == True:
        #batch_size = int(batch_size / 4)
        batch_size = int(batch_size / 6)
    elif augment_flip == True:
        batch_size = int(batch_size / 2)
    elif add_side_images == True:
        batch_size = int(batch_size / 3)
    print("batch_size:",batch_size)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = data_directory+'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #side images
                if (add_side_images == True):
                    # create adjusted steering measurements for the side camera images
                    #correction = 0.05 # this is a parameter to tune
                    correction = 0.2 # this is a parameter to tune
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction
                    left_name = data_directory+'IMG/'+batch_sample[1].split('/')[-1]
                    right_name = data_directory+'IMG/'+batch_sample[2].split('/')[-1]
                    left_image = cv2.imread(left_name)
                    right_image = cv2.imread(right_name)
                    #add images / angles
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(right_image)
                    angles.append(right_angle)
                #augmentation
                if (augment_flip == True):
                    images.append(cv2.flip(center_image,1))
                    angles.append(center_angle*-1.0)
                    if (add_side_images == True): #flip side images too
                        images.append(cv2.flip(left_image,1))
                        angles.append(left_angle*-1.0)
                        images.append(cv2.flip(right_image,1))
                        angles.append(right_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(len(X_train))
            yield sklearn.utils.shuffle(X_train, y_train)
        # if augment_flip == True:
        #     print("upa!")

def read_data(data_directory="./data/"):
    lines = []
    with open(data_directory+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers #skip header
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = data_directory+'IMG/' + filename
        #print(current_path)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def augment_images(images,measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    return np.array(augmented_images), np.array(augmented_measurements)

def get_simple_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))

    return model

def get_Lenet5_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x/ 255.0) - 0.5))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model

def get_nVidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


# setup data pipeline

# --- oringial pipeline loading in all data ---
#X_train, y_train = read_data()
#X_train, y_train = augment_images(X_train, y_train)
#print(len(X_train))
#print(len(y_train))
#print(X_train[0].shape)

# --- load using generators ---
sample_list = read_data_list()
#batch_size=32 # Set batch size
batch_size=36 # Set batch size
train_samples, validation_samples = train_test_split(sample_list, test_size=0.2)

# seperate generators for train / validate
train_generator = generator(train_samples, batch_size=batch_size, augment_flip = True, add_side_images = True)
#train_generator = generator(train_samples, batch_size=batch_size, augment_flip = True)
validation_generator = generator(validation_samples, batch_size=batch_size)
#input data width/height/channels
#channels, height, width = 3, 80, 320  # Trimmed image format
# setup model

model = get_nVidia_model() #get_Lenet5_model() #get_simple_model()

#train network
model.compile(loss = 'mse', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, epochs=5)
#model.fit(train_generator, steps_per_epoch= int(math.ceil(len(train_samples)/batch_size))*4,validation_data = validation_generator, validation_steps= int(math.ceil(len(validation_samples)/batch_size)), epochs=5, verbose = 1)
model.fit(train_generator, steps_per_epoch= int(math.ceil(len(train_samples)/batch_size))*6,validation_data = validation_generator, validation_steps= int(math.ceil(len(validation_samples)/batch_size)), epochs=5, verbose = 1)
#model.fit(train_generator, steps_per_epoch=300,validation_data = validation_generator, validation_steps=300, epochs=5, verbose = 1)

#save model
model.save('model.h5')
