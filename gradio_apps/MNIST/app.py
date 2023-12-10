# This requires gradio 3.50.2
# Use pip install gradio==3.50.2

import gradio as gr
import tensorflow as tf

# Load our model
model = tf.keras.models.load_model('model.keras')

def recognize_digit(image):
    if image is not None:
        image = image.reshape((1, 28, 28, 1)).astype('float32')/255

        prediction = model.predict(image)

        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return ''


# Build and launch the Gradio interface
iface = gr.Interface(
    fn = recognize_digit,
    inputs = gr.Image(
        shape=(28, 28), 
        image_mode='L', 
        invert_colors=True, 
        source='canvas',
        brush_radius=1, 
        tool="color-sketch",
    ),
    outputs = gr.Label(num_top_classes=3),
    live = True
).launch(share=True)