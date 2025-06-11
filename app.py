from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('melanoma_nevus_model.h5')

# Configuration (must match training)
ANATOM_SITE_CATEGORIES = ['head/neck', 'upper extremity', 'lower extremity',
                         'torso', 'palms/soles', 'oral/genital']
IMG_SIZE = (224, 224)
AGE_MIN = 10  # Replace with your actual min age from training
AGE_MAX = 90  # Replace with your actual max age from training

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    try:
        # Get form data
        file = request.files['image']
        sex = request.form.get('sex', 'male').lower() 
        age = request.form.get('age', '40')
        anatom_site = request.form.get('anatom_site', 'torso').lower()
        
        # Validate inputs
        if sex not in ['male', 'female']:
            return jsonify({'error': "Sex must be 'male' or 'female'"}), 400
            
        if anatom_site not in ANATOM_SITE_CATEGORIES:
            return jsonify({'error': f"Invalid anatomical site. Must be one of: {ANATOM_SITE_CATEGORIES}"}), 400
            
        try:
            age = float(age)
            if not (0 < age <= 120):
                raise ValueError
        except ValueError:
            return jsonify({'error': 'Age must be a number between 1 and 120'}), 400
        
        # Process image in memory
        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes)
        
        # Preprocess image
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess metadata
        sex_code = 0 if sex == 'male' else 1
        anatom_code = ANATOM_SITE_CATEGORIES.index(anatom_site)
        age_norm = (age - AGE_MIN) / (AGE_MAX - AGE_MIN)
        
        # Make prediction
        pred = model.predict([img_array,
                            np.array([[sex_code]]),
                            np.array([[anatom_code]]),
                            np.array([[age_norm]])])
        
        # Prepare response to match frontend expectations
        return jsonify({
            'diagnosis': {
                'Melanoma': float(pred[0][0]),
                'Nevus': float(pred[0][1])
            },
            'interpretation': interpret_prediction(pred[0])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def interpret_prediction(pred):
    mel_prob, nv_prob = pred
    if mel_prob > 0.7:
        return "High probability of melanoma - Consult a dermatologist immediately"
    elif mel_prob > 0.4:
        return "Moderate probability of melanoma - Recommended to see a specialist"
    else:
        return "Low probability of melanoma - Likely benign nevus"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)