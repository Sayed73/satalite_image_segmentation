import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tifffile as tiff
import io
import base64

app = Flask(__name__)

# Load the model
model = load_model(f'D:\Data Science\internships\Cellula\satalite project\Flask_app/UNet_Model.h5')

def read_image_tiff(image_file):
    return tiff.imread(image_file)

def preprocess_image(image_file):
    image_data = read_image_tiff(image_file)
    
    # Ensure the image has 12 channels
    if image_data.shape[2] != 12:
        raise ValueError("Input image must have 12 channels.")
    
    # Resize to 128x128 if necessary
    if image_data.shape[:2] != (128, 128):
        resized_image = np.zeros((128, 128, 12), dtype=image_data.dtype)
        for i in range(12):
            resized_image[:,:,i] = np.array(Image.fromarray(image_data[:,:,i]).resize((128, 128)))
        image_data = resized_image
    
    # Normalize the image
    image_data = image_data.astype(np.float32) / np.max(image_data)
    
    return np.expand_dims(image_data, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Save the file temporarily
            temp_path = os.path.join('temp', file.filename)
            file.save(temp_path)
            
            try:
                # Preprocess the image
                input_image = preprocess_image(temp_path)
                
                # Make prediction
                prediction = model.predict(input_image)
                
                # Convert prediction to binary mask
                binary_mask = (prediction > 0.5).astype(np.uint8) * 255
                
                # Convert input image and mask to base64 for display
                input_img = Image.fromarray((input_image[0, :, :, :3] * 255).astype(np.uint8))
                mask_img = Image.fromarray(binary_mask[0, ..., 0])
                
                buffered = io.BytesIO()
                input_img.save(buffered, format="PNG")
                input_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                buffered = io.BytesIO()
                mask_img.save(buffered, format="PNG")
                mask_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                return jsonify({
                    'input_image': input_base64,
                    'mask': mask_base64
                })
            except Exception as e:
                return jsonify({'error': str(e)})
            finally:
                # Clean up temporary file
                os.remove(temp_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(debug=True)