import flask_cors
from flask import Flask, request, app, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from werkzeug.utils import secure_filename
import json
import sklearn


app = Flask(__name__)
model = pickle.load(open('RandomForestClassifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_forestfire', methods=['POST','GET'])
def predict_forestfire():

    predict_data = request.json['data']
    print(predict_data.values())
    new_data = [list(predict_data.values())]
    output = model.predict(new_data)[0]
    return jsonify(str(output))

@app.route('/predict', methods=['POST','GET'])
def predict():
    predict_data = [float(x) for x in request.form.values()]
    final_features = [np.array(predict_data)]
    print(predict_data)
    output = model.predict(final_features)[0]
    if output==1:
        return render_template('home.html', prediction_text = 'Fire Forest Prediction: {}'.format("FIRE"))
    return render_template('home.html', prediction_text = 'Fire Forest Prediction: {}'.format("NO FIRE"))

# @app.route('/predict_batch', methods=['POST','GET'])
# def predict_batch():

#     if request.method=='POST':
#         f = request.files['file']
#         f.filename = "X_test.csv"
#         f.save("files/"+f.filename)
#         X_test = pd.read_csv('files/X_test.csv')
#         output = model.predict(X_test)
#         # for i in range(len(output)):
#         return render_template('batch.html', prediction_text = "{}".format(output))
        

if __name__=="__main__":
    app.run(debug=True)
