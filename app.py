# from flask import Flask, request, jsonify,render_template,send_file
# from PIL import Image
# import numpy as np
# import joblib
# import tensorflow as tf
# # import cv2

# app = Flask(__name__)

# # Function to load and preprocess image
# # def preprocess_image(image_path):
# #   image = cv2.imread(image_path) # original image
# #   # resizing image
# #   new_size = (400, 400)
# #   resized_image = cv2.resize(image, new_size)
# #   # converting into array and expanding dimensions
# #   img_array = np.expand_dims(resized_image, axis=0)
# #   return img_array
# print("TensorFlow version:", tf.__version__)
# def preprocess_image(image_path):
#     # Open image using Pillow
#     image = Image.open(image_path)

#     # Resizing image
#     new_size = (400, 400)
#     resized_image = image.resize(new_size)

#     # Converting image to numpy array and expanding dimensions
#     img_array = np.expand_dims(np.array(resized_image), axis=0)
#     return img_array

# # Load your model here
# # Example with TensorFlow/Keras
# import os
# # Get the current directory
# current_directory = os.getcwd()
# file_path = os.path.join(current_directory, "CNN_model.pickle")


# try:
#     # Load the model
#     model = joblib.load(file_path)
#     print("Model loaded successfully.")
# except Exception as e:
#     print("Error loading model:", e)
# # Construct the path to the classifier.pickle file

# # model = joblib.load(file_path)  # Make sure to have the correct path to your model file
# # model = tf.keras.models.load_model("model1.h5")
    
# model = joblib.load("_model.pickle")


# #label_encoder = joblib.load("label_encoder.pickle")
# CLASS_NAMES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     if file:
#         # Preprocess the image
#         img = preprocess_image(file)

#         # Make prediction using your model
#         pred = model.predict(img)
#         prediction = np.argmax(pred)
#         predicted_class = CLASS_NAMES[prediction]
#         print(predicted_class)


#         # Convert prediction to human-readable format (e.g., class labels)
#         # (This part depends on your model and its output format)
 
#         # Return the prediction
#         return jsonify({'predicted_class': predicted_class,'audio_url': '/music'})
# @app.route('/music')
# def get_audio():
#     # Return the audio file as a response
#     return send_file("country00012.png", mimetype="image/png", as_attachment=True, download_name="spectrogram.png")


# if __name__ == '__main__':
#     app.run(debug=True)








import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Load the saved model
model1 = load_model("model1.h5")

# Define class labels (replace with your actual class labels)
CLASS_NAMES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

@app.route('/', methods=['GET'])
def index():
    
    # Render the HTML file located in the "templates" folder
    return render_template('index.html')

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # original image
    # resizing image
    new_size = (400, 400)
    resized_image = cv2.resize(image, new_size)
    # converting into array and expanding dimensions
    img_array = np.expand_dims(resized_image, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the file to a temporary location
        image_path = 'temp_image.png'
        file.save(image_path)
        
        # Preprocess the image
        img_array = preprocess_image(image_path)
        
        # Make predictions
        predictions = model1.predict(img_array)
        predicted_label = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_label]
        
        os.remove(image_path)  # Remove the temporary image file
        
        return jsonify({'predicted_class': predicted_class})


@app.route('/predict/audio', methods=['POST'])
def predict_audio(): 
    
    return jsonify({'message': 'success'})
if __name__ == '__main__':
    app.run(debug=True)


