from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os
import numpy as np
from preprocess import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model('model.h5')

# Labels
class_names = ['cat', 'dog']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)

            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            confidence = round(float(np.max(prediction)) * 100, 2)

            result = class_names[class_index]

            return render_template('result.html',
                                   result=result,
                                   confidence=confidence,
                                   image_path=filepath)

    return render_template('index.html')

# ✅ VERY IMPORTANT
if __name__ == '__main__':
    app.run(debug=True)