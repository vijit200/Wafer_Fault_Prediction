
from flask import Flask , request , jsonify ,render_template
from prediction_validation_insertion import pred_validation
from PredictionFromModel import prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods = ['POST','GET'])
def predict():
    try:
        if request.method == 'POST':
            path = str(request.form['folder_path'])


            data = pred_validation(path)

            data.prediction_validation()

            tr = prediction(path)
            tr.predictionFromModel()

            return render_template('index.html',predict_wafer = 'Prediction File Created In Your Same Directory!....')
    except Exception as e:

        return render_template('index.html',predict_wafer = str(e))
        



if __name__ == '__main__':
    app.run(debug=True)


