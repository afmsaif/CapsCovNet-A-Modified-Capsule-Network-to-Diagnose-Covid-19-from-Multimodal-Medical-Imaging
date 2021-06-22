#loading dataset
import pickle
x = open('/content/gdrive/MyDrive/covidban/COVID-19 Radiography Database/Xray.pickle','rb')
x_train = pickle.load(x)
y = open('/content/gdrive/MyDrive/covidban/COVID-19 Radiography Database/yray.pickle','rb')
y_train = pickle.load(y)

#import libraries
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, Concatenate, LeakyReLU
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
plt.style.use('ggplot')

K.set_image_data_format('channels_last')
 
seed =142 


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1.0 + s_squared_norm)
    return (scale * x)

# capsule compatible batch_dot
def caps_batch_dot(x, y):
    x = K.expand_dims(x, 2)
    if K.int_shape(x)[3] is not None:
        y = K.permute_dimensions(y, (0,1,3,2))
    o = tf.matmul(x, y)
    return K.squeeze(o, 2)

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """
    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


def softmax(x, axis=-1):
    
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)

def pose_loss(y_true, y_pred):
    """.
    :param y_true: [None, n_classes, n_instance,pose]
    :param y_pred: [None, n_classes, n_instance,pose]
    :return: a scalar loss value.
    """
    return K.sum( K.square(y_true-y_pred),-1)


class Capsule(Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,
        
       
           
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(caps_batch_dot(c, hat_inputs))
            if i < self.routings - 1:
                b = caps_batch_dot(o, hat_inputs)
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

batch_size = 64
num_classes = 3
epochs = 40



filepath="weights-improvement-binary-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]



input_image = Input(shape=(128, 128, 3))

x1 = Conv2D(32, (7, 7), padding = 'valid', activation='relu')(input_image)
x1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x1)
x1 = Conv2D(64, (7, 7), padding = 'valid', activation='relu')(x1)
x1 = tf.keras.layers.ZeroPadding2D(padding=(8, 8))(x1)
x1 = AveragePooling2D((2, 2))(x1)
x1 = Conv2D(128, (7, 7), padding = 'valid', activation='relu')(x1)


#x1 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x1)

#x1 = Dropout(.5)(x1)
#x1 = Conv2D(128, (7, 7), padding = 'same', activation='relu')(x1)
##x1 = Conv2D(128, (3, 3), padding = 'same', activation='relu')(x1)

x2 = Conv2D(32, (5, 5), padding = 'valid',activation='relu')(input_image)
x2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', 
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                       beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x2)
#x2 = Conv2D(32, (5, 5),padding = 'same',activation='relu')(x2)
#x2 = AveragePooling2D((2, 2))(x2)
#x2 = Conv2D(64, (5, 5), padding = 'same',activation='relu')(x2)
x2 = Conv2D(64, (5, 5), padding = 'valid',activation='relu')(x2)
x2 = tf.keras.layers.ZeroPadding2D(padding=(4, 4))(x2)
x2 = AveragePooling2D((2, 2))(x2)
x2 = Dropout(.5)(x2)
x2 = Conv2D(128, (5, 5), padding = 'valid',activation='relu')(x2)


#x2 = Conv2D(128, (5, 5), padding = 'same',activation='relu')(x2)'''

x3 = Conv2D(32, (3, 3), padding = 'valid',activation='relu')(input_image)
x3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', 
                        gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x3)
#x3 = Conv2D(32, (3, 3), padding = 'same',activation='relu')(x3)
#x3 = AveragePooling2D((2, 2))(x3)
#x3 = Conv2D(64, (3, 3), padding = 'same',activation='relu')(x3)
x3 = Conv2D(64, (3, 3), padding = 'valid',activation='relu')(x3)
x3 = Dropout(.5)(x3)
x3 = AveragePooling2D((2, 2))(x3)
x3 = Conv2D(128, (3, 3), padding = 'valid',activation='relu')(x3)
#x3 = Conv2D(128, (3, 3), padding = 'same',activation='relu')(x3)


x = Concatenate()([x1, x2, x3])



x = Reshape((-1, 128))(x)
x4 = Capsule(32, 8, 3, True)(x)  
x5 = Capsule(32, 8, 3, True)(x4) 
x = Concatenate()([x4, x5])  
capsule = Capsule(3, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

mask_input= Input(shape=(num_classes, ))
mask = Mask()([capsule, mask_input])  # two inputs
dec = Dense(512, activation= tf.keras.layers.LeakyReLU(alpha=0.2))(mask)
dec = Dense(1024, activation= tf.keras.layers.LeakyReLU(alpha=0.2))(dec)
#dec = Dense(2048, activation='relu')(dec)
#dec = Dense(4096, activation='sigmoid')(dec)
dec = Dense(128*128*3, activation='sigmoid')(dec)
dec = Reshape(target_shape=[128, 128, 3])(dec)

catagorical = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0,
    name='categorical_crossentropy'
)
catagorical1 = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    name='sparse_categorical_crossentropy'
)



model = Model(inputs=[input_image,mask_input], outputs=[output,dec])

adam = optimizers.Adam(lr=0.001) 

#model.compile(loss=[margin_loss, 'mse'], loss_weights=[1., 0.5], optimizer=adam, metrics=[{'output':'accuracy'},{'dec': tf.keras.metrics.Accuracy()}])

model.compile(loss=[margin_loss, pose_loss], loss_weights=[1., 0.6], optimizer=adam, metrics=[{'output':'accuracy'},{'dec': tf.keras.metrics.Accuracy()}])

model.summary()


x_train1 = x_train
y_train1 = y_train

x_train1 = tf.cast(x_train1, dtype=tf.float32)
x_train1 = x_train1 / 255.0
  #x_train1 = tf.expand_dims(x_train1, axis = -1)
y_train1 = utils.to_categorical(y_train1, num_classes)
history = model.fit([x_train1, y_train1], [y_train1, x_train1],
        batch_size=batch_size,
        epochs=epochs,
        #validation_data=([x_valid1, y_valid1], [y_valid1, x_valid1]),
        validation_split = .1,
        #class_weight=class_weight_dic,
        callbacks=callbacks_list, 
        shuffle=True)   

model.save("model.h5")
print("Saved model to disk")
      
