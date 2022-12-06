import pickle
from flask import Flask, render_template, request, app, jsonify, url_for, redirect, escape
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

## Load the model
regModel = pickle.load(open('regModel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    dt = np.array(list(data.values())).reshape(1,-1)
    new_data=scalar.transform(dt)
    output=regModel.predict(new_data)
    # print(output[0])
    return jsonify(output[0].tolist())


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regModel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)