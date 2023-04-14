import cv2
import numpy as np
from Model import model
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the Keras model
input_shape = 480, 640, 3
models = model.DepthEstimationModel()
#
# models.build(input_shape=(None, 480, 640, 3))
# models.load_weights("/home/zahra/Hesam/depth estimation/weights/weights_unet.h5")
model = tf.keras.models.load_model("/home/zahra/Hesam/depth estimation/weights/model_unet")
# Create a VideoCapture object to capture video from the default camera (usually index 0)
# checkpoint = tf.train.Checkpoint(model=models)
# checkpoint.restore(tf.train.latest_checkpoint('/home/zahra/Hesam/depth estimation/checkpoint'))

cap = cv2.VideoCapture(-1)

# Check if the camera was successfully opened
if not cap.isOpened():
    print("Failed to open the camera")
    exit()

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()
#
#     # Check if the frame was successfully read
#     if not ret:
#         print("Failed to read a frame from the camera")
#         break
#
#     # Resize the frame to 480x640 pixels
#     input_image = cv2.resize(frame, (640, 480))
#     # Scale the pixel values to be between 0 and 1
#     # input_image /= 255
#
#     # Add a batch dimension to the input image
#     input_image = np.expand_dims(input_image, axis=0).astype("float32")
#
#     # Use the model to make a prediction on the input image
#     output_image = models.predict(input_image)
#     # Concatenate the input and output images horizontally
#     out = np.zeros((480, 640, 3)).astype("uint8")
#     out[:, :, 2:3] = output_image[0]
#     cmap = plt.cm.jet
#     cmap.set_bad(color="black")
#     concatenated_image = np.concatenate([frame, cmap(out)], axis=1)
#     # Show the concatenated image
#     cv2.imshow('Input and Output', concatenated_image)
#
#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()
cmap = plt.cm.jet
cmap.set_bad(color="black")
im = cv2.imread("/home/zahra/Hesam/test.JPG")
im = cv2.resize(im, (640,480)).astype("float32") / 255.0
im = tf.expand_dims(im,axis=0)
im = models.predict(im)
plt.imshow(im[0], cmap=cmap)
plt.show()
