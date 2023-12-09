import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Build the Neural Network.
model = Sequential(
    [
        Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), name ='L1'),
        MaxPooling2D((2,2), name='L2'),
        Conv2D(64, (3,3), activation='relu', name ='L3'),
        MaxPooling2D((2,2), name='L4'),
        Conv2D(64, (3,3), activation='relu', name ='L5'),
        Flatten(name='L6'),
        Dense(64, activation='relu', name='L7'),
        Dense(10, activation='softmax', name='L8')
    ]
)

# Compile the model.
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fit the model to the trainng data.
model.fit(
    train_images, train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

model.save('model.keras')