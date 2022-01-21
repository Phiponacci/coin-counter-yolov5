from cv2 import FONT_HERSHEY_SIMPLEX
from detect import run
import numpy as np
from flask import Flask, request, Response, jsonify
from PIL import Image
import cv2
from firebase_api import *
import io
import base64

app = Flask(__name__)


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf


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
        
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
        im = cv2.putText(im, label, (x1, y1), fontFace=FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,255,0), thickness=2)
        results.append(result)
    
    img = numpy_to_binary(im)
    saved_result = save_results(uid, img, results)
    return jsonify({"results": results, "image": saved_result['photo']})



if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
