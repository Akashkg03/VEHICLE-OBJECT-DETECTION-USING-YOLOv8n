from flask import Flask, request, render_template
import os
import torch
import numpy as np
import base64
from PIL import Image
from ultralytics import YOLO
import cv2  # Import OpenCV

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load the YOLOv8 model
model = YOLO('best.pt')

def infer_and_process(image_path):
    img = Image.open(image_path).convert('RGB')
    
    # Perform inference
    results = model(img)
    
    # Render predictions on the image
    img_with_boxes = results[0].plot()
    
    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(np.array(img_with_boxes), cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare predictions for display
    predictions = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        confidence = result.conf[0].item()
        class_id = int(result.cls[0].item())
        label = model.names[class_id]
        predictions.append({
            'class': label,
            'confidence': confidence
        })
    
    return img_str, predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith(('jpg', 'jpeg', 'png')):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img_str, predictions = infer_and_process(file_path)
            return render_template('result.html', image_data=img_str, predictions=predictions)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
