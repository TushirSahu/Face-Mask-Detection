from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D,Activation
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
# import numpy as np
model = Sequential([
    Conv2D(100, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(100, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
TRAINING_DIR='./train'
train_datagen=ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator=train_datagen.flow_from_directory(TRAINING_DIR,batch_size=10,target_size=(150,150))
VALID_DIR="./test"
validation_datagen=ImageDataGenerator(rescale=1.0/255)
validation_generator=validation_datagen.flow_from_directory(VALID_DIR,batch_size=10,target_size=(150,150))


checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit_generator(train_generator,epochs=8,validation_data=validation_generator,callbacks=[checkpoint])


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='training_accuracy')
plt.plot(history.history['val_accuracy'],label='validation_accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# print(model.evaluate(test_data,test_target))

