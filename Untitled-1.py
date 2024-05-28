# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential, datasets,models,layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dropout, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# %%
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


# %% [markdown]
# ### Load Cifar-10
# https://www.cs.toronto.edu/~kriz/cifar.html

# %%
from tensorflow.keras.datasets import cifar10

# %%
###Load both training data and test data from cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# %%
### Normalization for features of x
X_train = np.asarray(x_train, dtype=float)/255
X_test = np.asarray(x_test, dtype=float)/255

# %%
print(X_train.max())
print(X_test.max())
print(x_train.shape)


# %% [markdown]
# #### Show that Cifar-10 has been properly imported

# %%
fig = plt.figure(figsize=(8,8))
img = X_train[50]
plt.axis('on')
plt.imshow(img)
plt.title('Label: ' + str(y_train[50]))
plt.show()

# %%
fig = plt.figure(figsize=(8, 8))
columns = 10
rows = 10
for i in range(1, columns*rows +1):
    img = x_train[i]
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.title(y_train[i])
    plt.imshow(img)
plt.show()

# %% [markdown]
# According to the description of Cifar-10, the labels are encoded in to numbers, the corresponding labels are:
# 
# 0. airplane
# 1. automobile
# 2. bird
# 3. cat
# 4. deer
# 5. dog
# 6. frog
# 7. horse
# 8. ship
# 9. truck
# 
# ![屏幕截图 2024-04-12 162416.png](<attachment:屏幕截图 2024-04-12 162416.png>)

# %%
# x_train = x_train/225
# x_test = x_test/255

# %%
### Encode label data to one-hot format, with both 10 classes
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test,10)
print(y_cat_train[:5,:])
print(y_cat_test[:5, :])
print(y_train)

# %% [markdown]
# ### Build Model - Convolutional Neural Network

# %%
from keras.models import Sequential
def create_model():
    model_layers = [
        Conv2D(32, (3, 3), activation='relu', strides=(1,1), padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3,3), strides=(1, 1), padding='same', activation=keras.activations.relu, depth_multiplier=3),
    #     MaxPooling2D(2, 2),
        Dropout(rate =0.1),
        
        
        Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3,3), strides=(1, 1), padding='same', activation=keras.activations.relu),
    #     MaxPooling2D(2, 2),
        Dropout(rate = 0.1),
        
        Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3,3), strides=(1, 1), padding='same', activation=keras.activations.relu),
    #     MaxPooling2D(2, 2),
        Dropout(rate = 0.4),
        
        Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(1,1), strides=(1, 1), padding='same', activation=keras.activations.relu),
        
        
        Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same'),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(3,3), strides=(1, 1), padding='same', activation=keras.activations.relu),
        
        
        
        Conv2D(512, (1, 1), activation='relu', strides=(2, 2), padding='same'),
        BatchNormalization(),
        DepthwiseConv2D(kernel_size=(1,1), strides=(1, 1), padding='same', activation=keras.activations.relu),
        
    #     MaxPooling2D(2, 2),
        Dropout(rate = 0.4),
        
        Flatten(),
        Dropout(rate = 0.3),
        Dense(2048, activation='relu'),
        Dropout(rate = 0.3),
        Dense(512, activation='relu'),
        Dropout(rate = 0.4),
        Dense(10, activation='softmax')
    ] 
    model = Sequential(model_layers)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

# model = Sequential(model_layers)

# %%
model = create_model()
model.summary()

# %%
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# %%
### Monitor the validation loss on the validation set, and
### when no decrease takes place in 3 epochs, stop training.

early_stop = EarlyStopping(monitor='val_loss',patience=4)

# %% [markdown]
# #### Validation preparasion

# %%
X_train_, X_val, Y_train, Y_val = train_test_split(X_train, y_cat_train, random_state=10, test_size=0.3)
X_train_ = X_train[:24500]
Y_train = Y_train[:24500]
X_val = X_val[:10500]
Y_val = Y_val[:10500]
print(np.shape(X_train_))
print(np.shape(X_val))
print(np.shape(Y_train))
print(np.shape(Y_val))

# %% [markdown]
# ### MODEL FIT 1: No augmentation

# %%
model.fit(X_train_,Y_train,batch_size=64,epochs=20,validation_data=(X_val,Y_val),callbacks=[early_stop])

# %%
losses = pd.DataFrame(model.history.history)

# %%
losses[['loss','val_loss']].plot()

# %%
losses[['accuracy','val_accuracy']].plot()

# %%
print(model.metrics_names)
print(model.evaluate(X_test,y_cat_test,verbose=0))

# %%
predictions = np.argmax(model.predict(X_test), axis=-1)

# %%
print(classification_report(y_test,predictions))

# %%
print('y_train shape: ', np.shape(y_train))
print('y_test shape: ',np.shape(y_test))
print('predictions shape: ', np.shape(predictions))


# %%
confusion_matrix(y_test,predictions)

# %% [markdown]
# Elements on the diagonal represent the number of samples that the model correctly classifies.
# 
# Row i column j element: represents the number of samples whose true class is class i and the model predicts is class j.

# %%
### Play with specific samples:

classes = [0,1,2,3,4,5,6,7,8,9]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
d = dict(zip(classes, class_names))

# my_image_30 = x_test[30]
# my_image_35 = x_test[35]
# my_image_40 = x_test[40]
# my_image_45 = x_test[45]
# my_image_50 = x_test[50]
i_plot=1
for i in range (50, 71, 5):
    
    plt.subplot(1, 5, i_plot)
    i_plot+=1
    img = X_test[i]
    #fig.add_subplot(2, 5, i+1)
    plt.axis('off')
    plt.imshow(img)
    input_img = X_test[i].reshape(1,32,32,3)
    predictions = np.argmax(model.predict(input_img), axis=-1)[0]
    print(f"True class: {d[y_test[i][0]]} \nPredicted class: {d[predictions]}")
plt.show()




