from detect import run
import numpy as np
from flask import Flask, request, Response, jsonify
from PIL import Image
import cv2

app = Flask(__name__)


@app.route('/count', methods=['POST'])
def count():
    img = request.files['image']
    image = Image.open(img)
    im = np.array(image)
    im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    results = []
    preds = run(im)
    for pred in preds:
        # cents
        if pred[-1] >= 5:
            label = f"{pred[-1]} cents"
        else:
            # euros
            label = f"{pred[-1]} euros"
        result = {
            "confidence": pred[-2],
            "count": pred[-1],
            "label": label
        }
        results.append(result)
    
    return jsonify({"results": results})



if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
