import keras
from keras.models import Sequential # a linear stack of neural network layers
from keras.layers import Dense, Dropout, Flatten # Keras core layers (Activation)
from keras.layers import Conv2D, MaxPooling2D # Keras CNN layers (Convolution2D)
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import utils as Kutils
#from keras.regularizers import l2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

class trainer:

    path_dataset = 'English/Hnd/Img/Sample0'
    size = 36 # image size for learning
    model_name = ''
    with_pad = False
    with_fill = False
    params = {}
    num_class = 0
    
    ##################################################
    # Initialization                                 #
    ##################################################
    def __init__(self, params):
        self.model_name = params['model_name']
        if '_pad' in self.model_name : self.with_pad = True
        if '_fill' in self.model_name : self.with_fill = True
        self.params = params

    ##################################################
    # Train Dataset                                  #
    ##################################################
    def train_dataset(self):
    
        # load train & test datasets
        x_train,x_test,y_train,y_test = self.get_dataset()

        # data
        x_train = x_train.reshape(x_train.shape[0], self.size, self.size, 1)
        x_test = x_test.reshape(x_test.shape[0], self.size, self.size, 1)
        
        # label: convert vector to num_class matrices
        y_train = Kutils.to_categorical(y_train)
        y_test = Kutils.to_categorical(y_test)

        # layer params
        kernel_size = (3, 3) # filter kernel size
        activation = 'relu'
        padding = 'same' # input & output volumes (sizes) are same 
        input_shape = (self.size, self.size, 1)
        #kernel_regularizer = l2(0.0001) # 0.0001 - 0.001 (not much difference to use)
        strides = (2, 2) # step of the convolution
        #pool_size = (2, 2) # reducing the spatial dimensions (take the max value in the 2x2 filter)
        chan_dim = -1 # (x, y, z) = channels_last
        dropout = 0.25 # disconnection to the next layer to help generalization and avoid overfit

        # CNN (Convolutional Neural Network) model
        model = Sequential()

        # Conv Layers (the number of nodes (filters) in each layer depending on the size of the dataset)
        # defaults: strides=(1,1): each step, dilation_rate=1=(1,1): each gap, use_bias=True, kernel_initializer='glorot_uniform'        
        model.add(Conv2D(filters=32, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, input_shape=input_shape))
        model.add(BatchNormalization(axis=chan_dim))
        #model.add(MaxPooling2D(pool_size=(2, 2))) # not in use due to the accuracy rate
        
        model.add(Conv2D(filters=64, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
        model.add(BatchNormalization(axis=chan_dim))
        #model.add(MaxPooling2D(pool_size=(2, 2))) # not in use due to the accuracy rate
        
        model.add(Conv2D(filters=128, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation))
        model.add(BatchNormalization(axis=chan_dim))
        #model.add(MaxPooling2D(pool_size=(2, 2))) # becoming too small

        # Fully Connected Layer with 512 nodes
        model.add(Flatten())
        model.add(Dense(512, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        # Output Layer (softmax classifier = prediction values themselves)
        model.add(Dense(self.num_class, activation='softmax'))

        # params for compiling
        epochs = 20
        optimizer = keras.optimizers.Adam() # another optimizer: keras.optimizers.Adadelta()
        batch_size = 32

        # Compile the Model
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        # Data Augmentation (rotation, shift, shear, zoom)
        train_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, height_shift_range=0.08, shear_range=0.3, zoom_range=0.08)
        test_gen = ImageDataGenerator()
        train_idg = train_gen.flow(x_train, y_train, batch_size=batch_size)
        test_idg = test_gen.flow(x_test, y_test, batch_size=batch_size)
        hist = model.fit_generator(train_idg, steps_per_epoch=len(x_train)//batch_size, epochs=epochs, validation_data=test_idg, validation_steps=len(x_test)//batch_size)
        model.save(self.model_name)

        # Fit Model on the training data
        # hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

        print('The model has trained successfully.')

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        print(model.predict(x_test[:4]))
        print(y_test[:4])
        
        plt.style.use("ggplot")
        plt.figure()

        plt.plot(hist.history['accuracy'], label='train_accuracy')
        plt.plot(hist.history['val_accuracy'], label='val_accuracy')
        plt.plot(hist.history['loss'], label='train_loss')
        plt.plot(hist.history['val_loss'], label='val_loss')

        plt.title('Accuracy and Training Loss on Dataset')
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy / Loss')
        plt.legend(loc='lower left')
        plt.savefig('plot_' + self.model_name.replace('.h5', '') +'.png')        
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ##################################################
    # Get Dataset                                    #
    ##################################################
    def get_dataset(self):
        
        num_start = self.params['num_start']
        num_end = self.params['num_end']
        self.num_class = num_end-(num_start-1)
        num_data = self.params['num_data'] # number of dataset for each character
        num_train = self.params['num_train'] # number of training set
        num_test = num_data - num_train # number of test set
        
        x_train = np.zeros((self.num_class*num_train,self.size,self.size), np.float32)
        x_test = np.zeros((self.num_class*num_test,self.size,self.size), np.float32)
        y_train = np.zeros((self.num_class*num_train), np.uint8)
        y_test = np.zeros((self.num_class*num_test), np.uint8)

        if self.with_fill == True:
            xx_train = x_train.copy()
            xx_test = x_test.copy()
            yy_train = y_train.copy()
            yy_test = y_test.copy()
        
        x_train_cnt = 0
        x_test_cnt = 0

        for i in range(num_start, num_end+1):
            path_i = self.path_dataset
            num_i = ('0' if i<10 else '') + str(i)
            path_i += num_i + '/img0' + num_i + '-0'
            
            for j in range(1, num_data+1):
                num_j = ('0' if j<10 else '') + str(j)
                path = path_i + num_j + '.png'
                data,filled = self.refine_data(path)
                if j <= num_train:
                    x_train[x_train_cnt] = data
                    y_train[x_train_cnt] = i-num_start
                    if self.with_fill == True:
                        xx_train[x_train_cnt] = filled
                        yy_train[x_train_cnt] = i-num_start
                    x_train_cnt += 1
                else:
                    x_test[x_test_cnt] = data
                    y_test[x_test_cnt] = i-num_start
                    if self.with_fill == True:
                        xx_test[x_test_cnt] = filled
                        yy_test[x_test_cnt] = i-num_start
                    x_test_cnt += 1

        if self.with_fill == True:
            x_train = np.vstack((x_train, xx_train))
            x_test = np.vstack((x_test, xx_test))
            y_train = np.hstack((y_train, yy_train))
            y_test = np.hstack((y_test, yy_test))

        return x_train,x_test,y_train,y_test

    ##################################################
    # Refine Data                                    #
    ##################################################
    def refine_data(self, path):
        gray = cv2.imread(path, 0)
        inverse = np.where(gray<200, 1, 0).astype(np.uint8)
        im,filled = utils.refine(inverse, self.size, self.with_pad, self.with_fill)
        return im,filled
