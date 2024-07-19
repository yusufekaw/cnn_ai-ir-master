import numpy as np
from keras import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
from matplotlib import pyplot as plt


class Model_CNN_IR:
    model = None
    history = None
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    learning_rate = 0
    epochs = 0

    def __init__(self, x_train, y_train, x_test, y_test, learning_rate, epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def create_architecture(self):
        #persiapan input dimensi input dan output
        tensor_list = np.array(self.y_train)
        list = tensor_list.tolist()
        output_dim = len(set(list))
        input_shape = self.x_train[0].shape
        print("output shape ", output_dim)
        print("input shape ", input_shape)

        #create CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(24, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(12, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train_model(self):
        self.history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, verbose=1)

    def model_summary(self):
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True)
        print("---Model Image Saved--")

    def plot_Training(self):
        #plot training accuracy
        plt.plot(self.history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        #plot training loss
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

    def ModelPredict(self, data):
        predictions = self.model.predict(data, verbose=0)
        predicted_classes = tf.argmax(predictions, axis=1)
        return predicted_classes