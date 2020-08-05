from wsgiref import simple_server
from flask import Flask, request, app, jsonify, render_template
from flask import Response
from flask_cors import CORS
from logistic_deploy import predObj
import pandas as pd

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True


class ClientApi:

    def __init__(self):
        self.predObj = predObj()


@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
def predictRoute():
    try:
        # if request.json['data'] is not None:
        Pregnancies = float(request.form["Pregnancies"])
        Glucose = float(request.form["Glucose"])
        BloodPressure = float(request.form["BloodPressure"])
        SkinThickness = float(request.form["SkinThickness"])
        Insulin = float(request.form["Insulin"])
        BMI = float(request.form["BMI"])
        DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
        Age = float(request.form["Age"])
        # data = request.json['data']
        # print('data is:     ', data)

        dict_pred = {

            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age
        }

        data_df = pd.DataFrame(dict_pred, index=[1, ])

        pred = predObj()
        res = pred.predict_log(data_df)

        # result = clntApp.predObj.predict_log(data)
        print('result is        ', res)
        return render_template('results.html', res=res)
    # return jsonify({"Prediction": res})
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ', e)
        return Response(e)


if __name__ == "__main__":
    clntApp = ClientApi()
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    # httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
