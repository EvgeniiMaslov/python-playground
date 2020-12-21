from flask import Flask, Blueprint, render_template, request, url_for, redirect, flash
from werkzeug.utils import secure_filename
import os
from utils import wav_preprocess, write_song
import pickle
from xgboost import XGBClassifier
import numpy as np

ALLOWED_EXTENSIONS = ["wav"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "files"

loaded_model = pickle.load(open("model/model_mfcc.pkl", "rb"))
genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



@app.route('/', methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if 'file' not in request.files:
            flash('No file part')
            return render_template("index.html")

        if file.filename == '':
            flash('No selected file')
            return render_template("index.html")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            ceps = wav_preprocess(filename=filename, 
                                    file=file,
                                    method="mfcc",
                                    save=True, 
                                    save_path=app.config["UPLOAD_FOLDER"])
            
            ceps = np.reshape(ceps, (1, -1))

            
            prediction = loaded_model.predict(ceps)
            pred_class = genre_list[prediction[0]]

            return render_template("index.html", data=pred_class)
    else:
        return render_template("index.html")



if __name__ == '__main__':    
    app.run()