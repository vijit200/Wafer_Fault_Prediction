from prediction_validation_insertion import pred_validation
from PredictionFromModel import prediction
from flask import Flask ,render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])

def predict():
    if request.method == 'POST':
         path = str(request.form['folder_path'])

         data = pred_validation(path)

         data.prediction_validation()

         tr = prediction(path)
         tr.predictionFromModel()


         return render_template('index.html',predict_wafer = "Prediction file created !...")

if __name__ == '__main__':
    app.run(debug=True)