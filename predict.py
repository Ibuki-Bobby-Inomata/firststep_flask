import tensorflow as tf
keras = tf.keras
from flask import Flask, request, render_template
import numpy as np
# from keras.models import load_model
from datetime import datetime

import  preprocess.preprocesser as pre



app = Flask(__name__)

# global model, graph
model = keras.models.load_model("model/wave_model.h5")

@app.route("/", methods=["GET","POST"])
def upload_file():
    if request.method == "GET":
        return render_template("/index.html")

    if request.method == "POST":
        f = request.files["file"]
        filepath = "./data/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".csv"
        f.save(filepath)

        image_path = "./result_images/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"

        f1 = pre.creat_dataset(filepath, model, image_path)

        image_path = image_path.lstrip(".")

        # image_path = "/Users/bobby/Project/firstsetp_flask" + image_path

        print(image_path)

        return render_template("index.html", image_path=image_path, f1=f1)

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=5030)
