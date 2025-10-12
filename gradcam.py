import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv2d_2"):
    """
    Generate Grad-CAM heatmap for a CNN model.
    Args:
        model: Trained CNN model
        img_array: Preprocessed image (1, 128, 128, 3)
        last_conv_layer_name: Name of the last convolution layer in model
    Returns:
        heatmap: Grad-CAM heatmap (128x128)
    """
    # Create a model that maps the input image to activations & predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of class wrt conv output
    grads = tape.gradient(class_channel, conv_outputs)

    # Mean intensity of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    return superimposed_img
