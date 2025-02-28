# Smile Detector using the Deep learning
import os
from PIL import Image
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.optimizers import Adam

pixel_intensities = []
labels= []


directory = 'Training_data/'

for filename in os.listdir(directory):
    img= Image.open(directory+filename).convert('1')
    pixel_intensities.append(list(img.getdata()))

    if filename[0:5]=='happy':
        labels.append([1,0])
    elif filename[0:3] == 'sad':
        labels.append([0,1])
pixel = np.array(pixel_intensities)
pixel= pixel/255.0
labels = np.array(labels)

# print(pixel,labels)


model = Sequential()

model.add(Dense(1024,input_dim = 1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))

optimizer= Adam(learning_rate=0.005)

model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(pixel,labels,epochs=1000,batch_size=20)

print('Testing of Model.........')

test_image = Image.open('Test_data/happy_test.png').convert('1')
test_img= []
test_img.append(list(test_image.getdata()))
# print(test_img)
test_img1= np.array(test_img)
test_img1 = test_img1/255.0


print(model.predict(test_img1).round())