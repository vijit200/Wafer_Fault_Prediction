
from flask import Flask , request , jsonify ,render_template
from prediction_validation_insertion import pred_validation
from PredictionFromModel import prediction
from wsgiref import simple_server
import os

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

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
        



port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()


