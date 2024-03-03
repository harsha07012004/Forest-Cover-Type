#pip install flask
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Loading the mlr model
model = pickle.load(open('model.pkl', 'rb'))

# Flask is used for creating your application
# render template is used for rendering the HTML page
app = Flask(__name__)  # your application


@app.route('/')  # default route
def home():
    return render_template('index.html')  # rendering your home page.


@app.route('/pred', methods=['POST'])  # prediction route
def predict1():
    '''
    For rendering results on HTML 
    '''
    
    Elevation = request.form.get('elevation')
    Aspect = request.form.get('aspect')
    Slope = int(request.form.get('slope'))
    Horizontal_Distance_To_Hydrology = int(request.form.get('hordt-hyd'))
    Vertical_Distance_To_Hydrology = int(request.form.get('verdt-hyd'))
    Horizontal_Distance_To_Roadways = int(request.form.get('hordt-road'))
    Hillshade_9am = int(request.form.get('hillshade9'))
    Hillshade_Noon = int(request.form.get('hillshade12'))
    Hillshade_3pm = int(request.form.get('hillshade3'))
    Horizontal_Distance_To_Fire_Points = int(request.form.get('hordt-fire'))
    soil_type = request.form.get('soilType')

    # Create a DataFrame from the form data
    scaler = StandardScaler()
    numerical_features = [Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon,
                              Hillshade_3pm, Horizontal_Distance_To_Fire_Points]
    numerical_features_scaled = scaler.fit_transform([numerical_features])

        # Convert soil_type to a dataframe with 40 columns
    soil_type_list = list(soil_type)
    soil_type_df = pd.DataFrame([soil_type_list], columns=[f'Soil_Type{i}' for i in range(1, 41)])

        # Concatenate the numerical features and soil_type dataframes
    input_df = pd.DataFrame(np.concatenate([numerical_features_scaled, soil_type_df], axis=1))

        # Make prediction using the pre-trained model
    prediction = model.predict(input_df)
    print(prediction)

    outputs = ["Spruce/Fir","Lodgepole Pine","Ponderosa Pine","Cottonwood/Willow","Aspen","Douglas-fir","Krummholz"]

    return render_template("index.html", result="The predicted cover is " + outputs[prediction[0]-1]+"!")


# running your application
if __name__ == "__main__":
    app.run()