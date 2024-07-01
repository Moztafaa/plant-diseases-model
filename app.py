from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='tflite_qaware_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names
CLASS_NAMES = [
	"Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
	"Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
	"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
	"Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
	"Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
	"Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
	"Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
	"Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
	"Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
	"Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
	"Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
	"Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


@app.route('/predict', methods=['POST'])
def predict():
	if 'file' not in request.files:
		return jsonify({"error": "No file provided"}), 400

	file = request.files['file']
	if file.filename == '':
		return jsonify({"error": "No selected file"}), 400

	try:
		img = Image.open(file.stream)
		img = img.resize((128, 128))
		img_array = image.img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		img_array /= 255.0

		interpreter.set_tensor(input_details[0]['index'], img_array)
		interpreter.invoke()
		predictions = interpreter.get_tensor(output_details[0]['index'])

		predicted_class_index = np.argmax(predictions)
		predicted_class_name = CLASS_NAMES[predicted_class_index]
		confidence = predictions[0][predicted_class_index] * 100
		confidence = f"{confidence:.2f}%"  # Format the confidence to two decimal places
		return jsonify({"predicted_class": predicted_class_name, "confidence": confidence})
	except Exception as e:
		return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
	app.run(host="localhost", port=5000)
