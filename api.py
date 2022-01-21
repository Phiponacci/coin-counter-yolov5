from cv2 import FONT_HERSHEY_SIMPLEX
from detect import run
import numpy as np
from flask import Flask, request, Response, jsonify
from PIL import Image
import cv2
import io
import base64

app = Flask(__name__)



@app.route('/count', methods=['POST'])
def count():
    img = request.files['image']
    uid = request.form['uid']
    image = Image.open(img)
    im = np.array(image)
    im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    results = []
    preds = run(im)
    for pred in preds:
        x1, y1, x2, y2, confidence, classe = pred
        # cents
        if classe >= 5:
            label = f"{classe} cents"
        else:
            # euros
            label = f"{classe} euros"
        result = {
            "confidence": confidence,
            "label": label,
            "position": [x1, y1, x2, y2]
        }
        results.append(result)
    
    return jsonify({"results": results})



if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
