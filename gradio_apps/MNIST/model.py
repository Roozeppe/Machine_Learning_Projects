import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.models import Sequential

# Load, split and scale down the dataset features.
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = train_x.reshape((60000, 28, 28, 1)).astype('float32')/255
test_x = test_x.reshape((10000, 28, 28, 1)).astype('float32')/255

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

# Build the Neural Network.
model = Sequential(
    [
        Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), name='L1'),
        MaxPooling2D((2,2), name='L2'),
        Conv2D(64, (3,3), activation='relu', name='L3'),
        MaxPooling2D((2,2), name='L4'),
        Conv2D(64, (3,3), activation='relu', name='L5'),
        Flatten(name='L6'),
        Dense(64, activation='relu', name='L7'),
        Dense(10, activation='softmax', name='L8')
    ]
)

model.summary()

# Compile the model.
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model.
model.fit(
    train_x, train_y,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Save the model
model.save('model.keras')