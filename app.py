from flask import Flask, request, render_template, jsonify
import numpy as np
import json
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# creating flask app
app = Flask(__name__)




data_melbourne = {
    'Year': [1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033],
    'Production_Value_Barley': [696, 650.6, 897.9, 1116.3, 1386.2, 448.1, 1341.7, 1189.4, 928.3, 869.8, 1189.3, 1670.5, 1655.7, 478.4, 2274.5, 1305.1, 2003.6, 604.5, 1789.2, 1461.4, 1864.5, 1944.9, 2005.1, 1952.2, 2035.8, 1373.8, 1107.4, 3082.9, 2109.7, 1337.3, 3116.6, 2918.5, 2270.5, 2895.7, 2751.3, 1211.7, 2470.2, 2547.6, 1300.7, 2547.6, 2547.6, 1084.9, 2192.9, 1349.7, 2671.1],
    'Production_Value_Canola': [16.6, 9.5, 26.5, 23.5, 46.8, 56.7, 119.6, 131.6, 135.8, 257.4, 423.3, 379.7, 347.4, 176.7, 384.1, 342.5, 272.6, 42, 220.5, 233.2, 331.4, 476.2, 688.6, 866.2, 710, 558.7, 287.4, 632.7, 938, 510.7, 731.1, 1127.1, 1302.8, 1382.5, 1103.9, 322, 1102.5, 1099.6, 381.6, 1084.2, 1080.1, 412.9, 617.4, 452.2, 692.9],
    'Production_Value_Wheat': [1961.4, 1493, 1150.4, 2015.1, 2021.9, 933.6, 1921.2, 2262.3, 1502.8, 1462.3, 2642.1, 3079.7, 2791.4, 890.2, 3145.5, 1927.1, 2908.9, 879.3, 1995.3, 1755.7, 2994.9, 4412.4, 3943.3, 3422.9, 3395.9, 2631.3, 1814.9, 4664.8, 3682.1, 2276.6, 3714.3, 4525, 4246.4, 5392.9, 1103.9, 322, 1102.5, 1099.6, 381.6, 1084.2, 1080.1, 412.9, 617.4, 452.2, 692.9],
    'Precipitation (mm)': [354.4, 543.6, 489.4, 201.4, 217.8, 206.6, 168.8, 296.9, 273.8, 200.2, 291.7, 492.5, 154.8, 208.7, 244.5, 140.7, 191.1, 209.5, 327.2, 396.2, 341.3, 519.1, 399, 389.4, 364.2, 271.5, 135.5, 293, 395.1, 209, 240.1, 270.8, 258.2, 168.8, 266.1, 358.4, 264.3, 264.5, 314.2, 247.3, 234.2, 215.8, 226.3, 288.9, 267.7],
    'Yearly Mean Minimum Temperature': [8, 8.2, 7.7, 7.9, 8, 7.4, 7.3, 7.1, 7.2, 7.6, 8.4, 8.7, 8.1, 7.9, 8, 7.6, 8, 7.1, 8.8, 7.8, 8.5, 8.8, 8.4, 8, 8.4, 8.5, 8.1, 9, 8.3, 7.9, 8, 8.2, 8.2, 9, 8.5, 8.7, 8.4, 8.6, 7.9, 8.7, 7.9, 8.4, 8.5, 8.7],
    'Yearly Mean Maximum Temperature': [28.9, 29.2, 30.1, 27.9, 28.2, 30, 29, 27.7, 32.5, 30.4, 31.9, 31.5, 32, 28, 30.5, 30.2, 28.5, 31.7, 32.9, 31.2, 32.3, 30.8, 29.3, 29.5, 31.8, 31.6, 31, 29.8, 30.6, 31.4, 34.1, 29.9, 28.5, 31.5, 30.3, 31.8, 31.5, 30.1, 32, 30.6, 31, 32.8, 30.4, 32.6]
}

data_perth = {
    'Year': [1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
             2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'Production_Value_Barley': [900, 850.6, 1097.9, 1266.3, 1536.2, 588.1, 1441.7, 1289.4, 1028.3, 969.8, 900, 850.6,
                                1097.9, 1266.3, 1536.2, 588.1, 1441.7, 1289.4, 1028.3, 969.8, 900, 850.6, 1097.9, 1266.3,
                                1536.2, 588.1, 1441.7, 1289.4, 1028.3, 969.8, 900, 850.6],
    'Production_Value_Canola': [36.6, 29.5, 46.5, 43.5, 66.8, 76.7, 139.6, 151.6, 165.8, 287.4, 36.6, 29.5, 46.5, 43.5,
                                66.8, 76.7, 139.6, 151.6, 165.8, 287.4, 36.6, 29.5, 46.5, 43.5, 66.8, 76.7, 139.6, 151.6,
                                36.6, 29.5, 46.5, 43.5],
    'Production_Value_Wheat': [1915.1, 2121.9, 893.6, 1721.2, 2062.3, 1302.8, 1262.3, 1761.4, 1293,
                               1050.4, 1915.1, 2121.9, 893.6, 1721.2, 2062.3, 1302.8, 1262.3, 1761.4, 1293, 1050.4,
                               1915.1, 2121.9, 893.6, 1721.2, 2062.3, 1302.8, 1262.3, 1915.1, 2121.9, 893.6, 1721.2,
                               2062.3],
    'Precipitation (mm)': [374.4, 563.6, 509.4, 221.4, 237.8, 226.6, 188.8, 316.9, 293.8, 220.2, 374.4, 563.6, 509.4,
                           221.4, 237.8, 293.8, 220.2, 374.4, 563.6, 509.4, 221.4, 237.8, 226.6, 188.8, 316.9, 293.8,
                           220.2, 293.8, 220.2, 374.4, 563.6, 509.4],
    'Yearly Mean Minimum Temperature': [10, 9.5, 9.7, 9.9, 10, 9.4, 9.3, 9.1, 9.2, 9.6, 10, 9.5, 9.7, 9.9, 10, 9.4, 9.3,
                                        10, 9.5, 9.7, 9.9,
                                        10, 9.4, 9.3, 9.1, 9.2, 9.6, 9.9, 10, 9.4, 9.3, 9.1],
    'Yearly Mean Maximum Temperature': [29.9,
                                        30.4, 32.3, 31.1, 29.8, 34.5, 32.9, 30.5, 31.2, 32.1, 29.9, 30.4, 32.3, 31.1, 29.8,
                                        34.5, 32.9, 30.5, 31.2, 32.1, 29.9, 30.4, 32.3, 31.1, 29.8, 34.5, 32.9, 29.9, 30.4,
                                        32.3, 31.1, 29.8]
}
data_brisbane = {
    'Year': [1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
             2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'Production_Value_Barley': [750, 700.6, 897.9, 1066.3, 1336.2, 438.1, 1091.7, 989.4, 828.3, 769.8, 750, 700.6,
                                897.9, 1066.3, 1336.2, 438.1, 1091.7, 989.4, 828.3, 769.8, 750, 700.6, 897.9, 1066.3,
                                1336.2, 438.1, 1091.7, 989.4, 828.3, 769.8, 750, 700.6],
    'Production_Value_Canola': [46.6, 39.5, 86.7, 149.6, 161.6, 175.8, 297.4, 46.6, 39.5, 56.5, 53.5, 76.8, 86.7, 149.6, 161.6,
                                175.8, 297.4, 46.6, 39.5, 56.5, 53.5, 76.8, 86.7, 149.6, 161.6, 175.8, 297.4, 297.4,
                                46.6, 39.5, 56.5, 53.5],
    'Production_Value_Wheat': [1961.4, 1493, 1350.4, 2015.1, 2121.9, 993.6, 1721.2, 2062.3, 1302.8, 1262.3, 1961.4,
                               1493, 1350.4, 2015.1, 2121.9, 993.6, 1721.2, 2062.3, 1302.8, 1262.3, 1961.4, 1493,
                               1350.4, 1721.2, 2062.3, 1302.8, 1262.3, 2015.1, 2121.9, 993.6, 1721.2,
                               2062.3],
    'Precipitation (mm)': [354.4, 543.6, 489.4, 201.4, 217.8, 206.6, 168.8, 296.9, 273.8, 200.2, 354.4, 543.6, 489.4,
                           201.4, 296.9, 273.8, 200.2, 354.4, 543.6, 489.4, 201.4, 217.8, 206.6, 168.8, 296.9, 273.8,
                           200.2, 273.8, 200.2, 354.4, 543.6, 489.4],
    'Yearly Mean Minimum Temperature': [11,
                                        10.4, 10.3, 10.1, 10.2, 10.6, 11, 10.5, 10.7, 10.9, 11, 10.4, 10.3, 10.1, 10.2,
                                        10.6, 11, 10.5, 10.7, 10.9, 11, 10.4, 10.3, 10.1, 10.2, 10.6, 10.9, 11, 10.4,
                                        10.3, 10.1],
    'Yearly Mean Maximum Temperature': [31.5, 32.2, 33.1, 30.9, 31.4, 33.3, 32.1, 30.8, 35.5, 33.9, 31.5, 32.2, 33.1, 30.9,
                                        30.8,
                                        35.5, 33.9, 31.5, 32.2, 33.1, 30.9, 31.4, 33.3, 32.1, 30.8, 35.5, 33.9, 30.9, 31.4,
                                        33.3, 32.1, 30.8]
}



@app.route('/')
def index():
    return render_template("index.html")


# Route to render the visualization page
@app.route('/visualization')
def visualization():
    return render_template("visualization.html")


@app.route('/get_data', methods=['POST'])
def get_data():
    selected_location = request.form['location']
    selected_crop = request.form['crop']

    # Retrieve data based on location
    if selected_location == "Melbourne":
        data = data_melbourne
    elif selected_location == "Perth":
        data = data_perth
    elif selected_location == "Brisbane":
        data = data_brisbane
    else:
        return jsonify({'error': 'Invalid location'})

    # Check if the selected crop is valid
    if selected_crop in data:
        selected_data = data[selected_crop]
    else:
        return jsonify({'error': 'Invalid crop'})

    # Include temperature data in the response
    temperature_data = data['Yearly Mean Maximum Temperature']

    # Create a dictionary with both crop and temperature data
    data_dict = {
        'Year': data['Year'],
        selected_crop: selected_data,
        'Yearly Mean Maximum Temperature': temperature_data
    }

    return json.dumps(data_dict)  # Serialize data_dict to JSON


@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosphorus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the ideal crop for the given climate conditions.".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html', result=result, scroll_to_result=True)

    # Return a JSON response
    #return jsonify({"result": result})


# python main
if __name__ == "__main__":
    app.run(debug=True)
