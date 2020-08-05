import pickle
import numpy as np
import pandas as pd
from logistic_deploy import predObj

scalar = pickle.load(open("sandardScalar.sav", 'rb'))
# with open("standardScalar.sav", 'rb') as f:
#     scalar = pickle.load(f)
model = pickle.load(open("modelForPrediction.sav", 'rb'))
# with open("modelForPrediction.sav", 'rb') as f:
#     model = pickle.load(f)

dict_pred = {

    "Pregnancies": 0.022,
    "Glucose": 0.009,
    "BloodPressure": 0.005,
    "SkinThickness": 0.006,
    "Insulin": 0.012,
    "BMI": 0.45,
    "DiabetesPedigreeFunction": 0.45456,
    "Age": 0.345
}

data_df = pd.DataFrame(dict_pred, index=[1, ])

# print(data_df)
scaled_data = scalar.transform(data_df)
predict = model.predict(scaled_data)
if predict[0] == 1:
    print('Diabetic')
else:
    print('Non-Diabetic')
