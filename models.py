import keras
from keras import Model, Input
#from keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
#from keras.utils import plot_model
from keras.callbacks import *
from keras.layers import *
import tensorflow as tf

def channelAttention(inLayer, n):
  att_1 = GlobalAveragePooling2D()(inLayer)
  att_1 = Dense(att_1.shape[1]*n, activation = 'relu')(att_1)
  att_1 = Dense(int(att_1.shape[1]/n), activation = 'sigmoid')(att_1)
  out = Multiply()([inLayer, tf.expand_dims(tf.expand_dims(att_1,axis=1), axis=1)])
  return out
  
  
  
  
  
def pixelAttention(inLayer, n):
  att_1 = Conv2DTranspose(filters = n*inLayer.shape[3], kernel_size = 3, strides = n, activation = 'relu')(inLayer)
  att_1 = BatchNormalization()(att_1)
  print(att_1.shape)
  att_1 = Conv2D(filters = inLayer.shape[3], kernel_size = 3, strides = n, activation = 'sigmoid')(att_1)
  att_1 = BatchNormalization()(att_1)
  print(att_1.shape)
  print(inLayer.shape)
  out = Multiply()([inLayer, att_1])
  return out
  
  
  
  
  
def network (height = processed_X_train.shape[1], width = processed_X_train.shape[2], channels = processed_X_train.shape[3]):
  inputs = Input((height, width, channels))
  mid_1 = Conv2D(filters = 32, kernel_size = 3, strides= 1, padding = 'valid')(inputs)
  out1 = pixelAttention(mid_1, 2)
  out1 = BatchNormalization()(out1)
  print(out1.shape)
  out1 = Conv2D(filters = 64, kernel_size = 3, strides= 1, padding = 'valid')(out1)
  out1 = BatchNormalization()(out1)
  out1 = GlobalAveragePooling2D()(out1)
  out1 = Dense(64, activation = 'softmax')(out1)

  conv1 = Conv1D(filters = 4, kernel_size= 3, strides= 2, padding='valid', activation='relu')(inputs)
  conv1 = BatchNormalization()(conv1)
  print(conv1.shape)
  conv2 = Conv1D(filters = 8, kernel_size= 3, strides= 2, padding='valid', activation='relu')(conv1)
  conv2 = BatchNormalization()(conv2)
  print(conv2.shape)
  conv3 = Conv1D(filters = 16, kernel_size= 3, strides= 2, padding='valid', activation= 'relu')(conv2)
  conv3 = BatchNormalization()(conv3)
  print(conv3.shape)
  conv4 = Conv1D(filters = 32, kernel_size= 3, strides= 2, padding='valid', activation = 'relu')(conv3)
  conv4 = BatchNormalization()(conv4)
  print(conv4.shape)
  
  '''
  reshaped = Reshape((493, 32, 1))(conv4)
  print(reshaped.shape)

  conv5 = Conv2D(filters = 64, kernel_size=3, strides = 2, padding='valid', activation = 'relu')(reshaped)
  print(conv5.shape)

  conv6 = Conv2D(filters = 128, kernel_size=3, strides = 2, padding='valid', activation = 'relu')(conv5)
  print(conv6.shape)

  '''
  flat = Flatten()(conv4)
  print(flat.shape)

  den1 = Dense(64, activation = 'softmax')(flat)
  print(den1.shape)
  mul = Concatenate()([den1, out1])
  mul = BatchNormalization()(mul)
  out = Dense(1, activation = 'sigmoid')(mul)
  model = Model(inputs = inputs, outputs = out)

  return model



def architecture_1(height = processed_X_train.shape[1], width = processed_X_train.shape[2], channels = processed_X_train.shape[3]):
  inputs = Input((height, width, channels))
  conv1 = Conv1D(filters = 4, kernel_size= 3, strides= 2, padding='valid', activation='relu')(inputs)
  conv1 = BatchNormalization()(conv1)
  print(conv1.shape)
  conv2 = Conv1D(filters = 8, kernel_size= 3, strides= 2, padding='valid', activation='relu')(conv1)
  conv2 = BatchNormalization()(conv2)
  print(conv2.shape)
  conv3 = Conv1D(filters = 16, kernel_size= 3, strides= 2, padding='valid', activation= 'relu')(conv2)
  conv3 = BatchNormalization()(conv3)
  print(conv3.shape)
  conv4 = Conv1D(filters = 32, kernel_size= 3, strides= 2, padding='valid', activation = 'relu')(conv3)
  conv4 = BatchNormalization()(conv4)
  print(conv4.shape)

  flat = Flatten()(conv4)
  print(flat.shape)

  den1 = Dense(6, activation = 'softmax')(flat)
  model = Model(inputs = inputs, outputs = den1)

  return model



def architecture_2(height = processed_X_train.shape[1], width = processed_X_train.shape[2], channels = processed_X_train.shape[3]):
  inputs = Input((height, width, channels))
  mid_1 = Conv2D(filters = 32, kernel_size = 3, strides= 1, padding = 'valid')(inputs)
  out1 = BatchNormalization()(mid_1)
  print(out1.shape)
  out1 = Conv2D(filters = 64, kernel_size = 3, strides= 1, padding = 'valid')(out1)
  out1 = BatchNormalization()(out1)
  out1 = GlobalAveragePooling2D()(out1)
  out1 = Dense(6, activation = 'softmax')(out1)
  model = Model(inputs = inputs, outputs = out1)

  return model
  
  
def architecture_3(height = processed_X_train.shape[1], width = processed_X_train.shape[2], channels = processed_X_train.shape[3]):
  inputs = Input((height, width, channels))
  mid_1 = Conv2D(filters = 32, kernel_size = 3, strides= 1, padding = 'valid')(inputs)
  out1 = pixelAttention(mid_1, 2)
  out1 = BatchNormalization()(out1)
  print(out1.shape)
  out1 = Conv2D(filters = 64, kernel_size = 3, strides= 1, padding = 'valid')(out1)
  out1 = BatchNormalization()(out1)
  out1 = GlobalAveragePooling2D()(out1)
  out1 = Dense(6, activation = 'softmax')(out1)

  model = Model(inputs = inputs, outputs = out1)

  return model




