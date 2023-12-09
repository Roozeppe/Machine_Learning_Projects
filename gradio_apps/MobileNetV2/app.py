import tensorflow as tf
import numpy as np

import gradio as gr

inception_net = tf.keras.applications.MobileNetV2()

# Open categorical labels for reading.
file_obj = open('labels.txt', 'r')

response = file_obj.read()
labels = response.splitlines()

file_obj.close()

def classify_image(inp):
  
  # Resize the image.
  width = abs(inp.size[0] - (inp.size[0] - 224))
  height = abs(inp.size[1] - (inp.size[1] - 224))

  inp = inp.resize((round(width), round(height)))

  # Convert the input PIL image into a numpy array.
  inp = np.array(inp)

  inp = inp.reshape((-1, 224, 224, 3))
  inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
  prediction = inception_net.predict(inp).flatten()
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences

# Build the Gradio interface
gr.Interface(fn=classify_image,
             inputs=gr.Image(type='pil'),
             outputs=gr.Label(num_top_classes=3),
            ).launch(share=True)