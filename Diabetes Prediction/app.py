import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Specify the directory where the template files are located
template_folder = os.path.join(os.path.expanduser('~'), 'python-ws')

# Set the template folder for the Flask app
app.template_folder = template_folder

# Specify the directory where the pickle files are located
pickle_folder = os.path.join(os.path.expanduser('~'), 'python-ws')

# Load the scaler and model from the pickle files
sc = pickle.load(open(os.path.join(pickle_folder, 'sc.pkl'), 'rb'))
model = pickle.load(open(os.path.join(pickle_folder, 'classifier.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('ditect.html')

@app.route('/',methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model.predict(sc.transform(final_features))
    return render_template('result.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
