from flask import Flask, request, jsonify, render_template
import joblib
import requests
import pandas as pd
import datetime
import numpy as np

app = Flask(__name__)

#Carga de modelos
models = {
'RandomForest' : joblib.load('../3 Entrenamiento/Random Forest/random_forest15_1.pkl'),
'XGBoost' : joblib.load('../3 Entrenamiento/XGBoost/XGBoost15_1.pkl'),
'LSTM' : joblib.load('../3 Entrenamiento/LSTM/LSTM15_1.pkl'),
'LSTMBir': joblib.load('../3 Entrenamiento/LSTMBir/LSTM Bir15_1.pkl'),
}
#Variable global para guardar el resultado de llamar a la api
predictionData = pd.DataFrame()
#Obtiene los datos de los últimos 15 días de las api de AEMET y de la CAM
def getPredictionData():
    global predictionData
    #Dia para el que se quiere predecir el polen al día siguiente
    # hoy = datetime.date(2024,5,15) #Este día funciona porque hay suficientes datos de días anteriores, es posible que fechas futuras fallen
    hoy = datetime.date.today() #Si la aplicación no funciona comentar esta línea y descomentar la anterior
    # hoy = datetime.date(2024,5,1) #Este día ofrece predicciones en alto, medio y bajo dependiendo del modelo elegido
    
    #15 días antes
    fechaInicial = hoy - datetime.timedelta(days=15)
    #Request de los dato meteorológicos
    headers = {
        'accept': 'application/json',
        'api_key': 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJpdWxpdXNnaEB1Y20uZXMiLCJqdGkiOiI0MGY5NjI2NC1mOWQ0LTQ2MGEtOWEzOS00YTA4NTgwNTZiYTciLCJpc3MiOiJBRU1FVCIsImlhdCI6MTcxMjEzNjE2OSwidXNlcklkIjoiNDBmOTYyNjQtZjlkNC00NjBhLTlhMzktNGEwODU4MDU2YmE3Iiwicm9sZSI6IiJ9.Cwqe8bk7jiodRJXg-77p1VA3DKRM1a1Tl23o9e2Iu4Q',
    }
    response = requests.get(
        'https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/'+str(fechaInicial)+'T00%3A00%3A00UTC/fechafin/'+str(hoy) +'T00%3A00%3A00UTC/estacion/3200',
        headers=headers,
    )
    #Se cargan los datos meteorológicos de los últimos 15 días en un dataframe
    meteo = pd.json_normalize(requests.get(response.json().get('datos')).json())
    #Se procesan los datos para ponerlos en un formato adecuado
    meteo['fecha']=pd.to_datetime(meteo['fecha'])
    meteo['prec']=meteo['prec'].apply(lambda x: '0,0' if x == 'Ip' else x)
    meteo["prec"] = meteo["prec"].astype(str).str.replace(",", ".").astype(float)
    meteo["tmin"] = meteo["tmin"].astype(str).str.replace(",", ".").astype(float)
    meteo["tmax"] = meteo["tmax"].astype(str).str.replace(",", ".").astype(float)
    meteo["dir"] = meteo["dir"].astype(float)
    meteo["velmedia"] = meteo["velmedia"].astype(str).str.replace(",", ".").astype(float)
    meteo["racha"] = meteo["racha"].astype(str).str.replace(",", ".").astype(float)
    meteo["sol"] = meteo["sol"].astype(str).str.replace(",", ".").astype(float)
    #request para los datos de polen
    polen = requests.get(
        "https://datos.comunidad.madrid/catalogo/api/action/datastore_search?id=1eeed6ac-32cc-4cf1-b976-52bea51ab964&fields=captador,fecha_lectura,tipo_polinico,granos_de_polen_x_metro_cubico&limit=40000"
    )
    #Se cargan los datos de polen en un dataframe
    df = pd.json_normalize(polen.json().get('result').get('records'))
    #Se procesan los datos para ponerlos en un formato adecuado
    df['fecha']=pd.to_datetime(df['fecha_lectura'])
    df=df.reset_index(drop=True)
    polen=df[(df['captador']=='GETA') & (df['tipo_polinico'] == 'Gramíneas') & (df['fecha'].dt.date >= fechaInicial)]
    polen = polen.sort_values(by='fecha')
    polen['granos_de_polen'] = polen['granos_de_polen_x_metro_cubico'].ffill()
    #Se unen los dos dataframes
    merged =  pd.merge(meteo,polen,on='fecha')
    #Columnas con datos importantes
    columnas=['granos_de_polen', 'prec', 'tmin', 'tmax', 'dir', 'velmedia', 'racha', 'sol','fecha']
    #Se crea un nuevo dataframe sol con las columnas relevantes
    datos=pd.DataFrame(columns=columnas)
    for columna in columnas :
        datos[columna]=merged[columna]
    #Se adapta el dataframe al formato de entrada del modelo
    for i in range(1, 16):
        for columna in ['granos_de_polen', 'prec', 'tmin', 'tmax', 'dir', 'velmedia', 'racha', 'sol']:
            # Supress SettingWithCopyWarning
            datos = datos.copy()
            datos[f'{columna}_{i}'] = datos[columna].shift(i)
    datos['semana'] = datos['fecha'].dt.dayofyear // 7
    datos['mes'] = datos['fecha'].dt.month
    datos['dia'] = datos['fecha'].dt.dayofyear
    datos=datos.drop(columns=['fecha'])
    predictionData = datos.iloc[-1:]
#Adapta el formato a la entrada del RandomForest
def randomForestFormat(data):
    returnData = data
    returnData=returnData.drop(columns=['granos_de_polen','dia','mes','semana'])
    return returnData.iloc[-1:]
#Adapta el formato a la entrada del XGBoost
def XGBoostFormat(data):
    returnData=data
    returnData=returnData.drop(columns=['dia','mes','semana'])
    return returnData.iloc[-1:]
#Adapta los datos a la entrada de las LSTM
def LSTMFormat(data):
    #Se carga el scaler utilizado en el entrenamiento de la LSTM
    scaler = joblib.load('scaler/LSTM15_1_scaler.pkl')
    returnData = data
    returnData=returnData.drop(columns=['granos_de_polen'])
    transformed_data = scaler.transform(returnData)
    array = np.array(transformed_data)
    reshaped = array.reshape((array.shape[0], 1, array.shape[1]))
    return reshaped[-1:]
#Transforma el resultado numérico a categórico
def predictionToCategory(prediction):
    if np.isnan(prediction):
        return "Nulo"

    result=""
    if prediction < 25:
        result = "Bajo"
    elif prediction < 50:
        result = "Medio"
    else:
        result = "Alto"
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global predictionData
    # Get data from POST request
    data = request.json
    model_name=data['model']
    if model_name != 'All':
        model=models.get(model_name)
        if model is None:
            return jsonify({'error': 'Model not found'}), 400
        #Se adapta el formato a la entrada de cada modelo
        if model_name == 'RandomForest':
            predictionDataFormatted=randomForestFormat(predictionData)
        elif model_name == 'XGBoost':
            predictionDataFormatted=XGBoostFormat(predictionData)
        elif model_name == 'LSTM' or model_name == 'LSTMBir':
            predictionDataFormatted=LSTMFormat(predictionData)
        #Predicción
        prediction = model.predict(predictionDataFormatted)
        #Devuelve la predicción en texto
        pred_str=""
        if model_name == 'LSTM' or model_name == 'LSTMBir':
            pred_str=str(round(prediction[0][0], 2))
        else:
            pred_str=str(round(prediction[0], 2))
        return jsonify({'prediction': f'Utilizando el modelo {model_name} se ha obtenido\
                         una predicción de {pred_str} granos de polen por metro cúbico lo\
                           cual es un valor de polen {predictionToCategory(prediction[0])}.'})
    else:
        rf=models.get('RandomForest')
        xgb=models.get('XGBoost')
        lstm=models.get('LSTM')
        lstmB=models.get('LSTMBir')
        prediction1 = predictionData
        prediction2 = prediction1
        prediction3 =  prediction1
        prediction4 = prediction1
        prediction1 = randomForestFormat(prediction1)
        prediction2 = XGBoostFormat(prediction2)
        prediction3 = LSTMFormat(prediction3)
        prediction4 = LSTMFormat(prediction4)
        prediction_mean = (rf.predict(prediction1)[0] + xgb.predict(prediction2)[0] + lstm.predict(prediction3)[0][0] + lstmB.predict(prediction4)[0][0]) / float(len(models))
        return jsonify({'prediction' : f'La predicción media de todos los modelos es\
                         {str(round(prediction_mean, 2))} granos de polen por metro cúbico\
                              lo cual es un valor de polen {predictionToCategory(prediction_mean)}.'})



if __name__ == '__main__':
    getPredictionData()
    app.run(debug=True)
    
 
