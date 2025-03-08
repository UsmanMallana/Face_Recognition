import tensorflow as tf
from tensorflow.keras import layers, models
import h5py
import numpy as np

class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def build_replacement_model():
    input_layer = layers.Input(shape=(128, 128, 3), name="input_1")

    x = layers.Conv2D(32, (3,3), strides=1, padding='same', name="conv2d")(input_layer)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Activation('relu', name="activation")(x)
    x = layers.MaxPooling2D((2,2), name="max_pooling2d")(x)

    x = layers.Conv2D(64, (3,3), strides=1, padding='same', name="conv2d_1")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.Activation('relu', name="activation_1")(x)
    x = layers.MaxPooling2D((2,2), name="max_pooling2d_1")(x)

    x = layers.DepthwiseConv2D((3,3), strides=1, padding='same', name="depthwise_conv2d")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.Activation('relu', name="activation_2")(x)
    x = layers.MaxPooling2D((2,2), name="max_pooling2d_2")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation='relu', name="dense")(x)
    x = layers.Dense(128, activation='linear', name="dense_1")(x)

    # Replace the old Lambda with a custom layer
    output_layer = L2NormalizeLayer()(x)

    new_model = models.Model(inputs=input_layer, outputs=output_layer, name="face_model_fixed")
    return new_model

def load_weights_from_h5(old_h5_path, new_model):
    """
    Manually read the layer weights from the old H5 file
    and assign them to 'new_model' layers.
    """
    with h5py.File(old_h5_path, "r") as f:
        # Typically, weights are under "model_weights"
        if "model_weights" not in f:
            raise ValueError("No 'model_weights' group found in old H5 file.")

        weights_group = f["model_weights"]

        # Iterate over new_model layers, skipping the final custom layer if it has no weights
        for layer in new_model.layers:
            if layer.name in weights_group:
                layer_weights = []
                layer_group = weights_group[layer.name]

                # Each layer has an attribute "weight_names" listing its parameter names
                weight_names = layer_group.attrs["weight_names"]
                for wn in weight_names:
                    weight_val = layer_group[wn][:]
                    layer_weights.append(weight_val)

                # Assign these weights to the current layer
                layer.set_weights(layer_weights)
            else:
                # It's okay if the final custom layer is missing from old weights
                print(f"Skipping weights for layer: {layer.name}")
# 1. Build the new model
new_face_model = build_replacement_model()
new_face_model.summary()

# 2. Copy weights from the old file
OLD_H5_FILE = "face_recognition.h5"  # Path to your old (unloadable) model
load_weights_from_h5(OLD_H5_FILE, new_face_model)

# 3. (Optional) compile if you plan to do more training
new_face_model.compile(optimizer="adam")
new_face_model.save("fixed_model.h5")