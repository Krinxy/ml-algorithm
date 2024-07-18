import numpy as np
import tensorflow as tf  


features = np.zeros((10, 10, 10))
labels = np.random.randint(0, 2, size=(10,))

for i in range(10):
    for j in range(10):
        dimension = np.random.randint(0, 10)
        features[i, j, :] = dimension

print(features)
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, input_shape=(features.shape[1], features.shape[2]), name='lstm_layer'))
model.add(tf.keras.layers.Dense(64, activation='relu', name='intermediate_dense'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.25))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax', name='output_layer'))  # classes

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(
                  learning_rate=0.001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-07), metrics=['accuracy'])
model.fit(features, labels, epochs=150, batch_size=10, shuffle=True)

model.summary()
