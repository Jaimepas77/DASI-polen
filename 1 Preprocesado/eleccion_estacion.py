# -*- coding: utf-8 -*-
"""
Se ha realizado un estudio de la completitud de los datos de las diferentes estaciones meteorológicas 
de Madrid para elegir la que tuviera los datos más completos. De esta manera se busca obtener los mejores resultados
para los modelos predictivos.
"""

import pandas as pd
import os
import numpy as np

#columnas importantes para la predicción
columnas = ['filename','prec','tmin','tmax','dir','velmedia','racha','sol','total']
#se crea un dataframe con las columnas importantes
result = pd.DataFrame(columns=columnas)
#ruta a la carpeta que contiene todos los ficheros json de las distintas estaciones meteorologicas
path = '../0 data/meteorologia'
#Para cada fichero
for filename in os.listdir(path):
    #Se abre el archivo
    with open(os.path.join(path,filename), 'r') as f:
        #Se cargan los valores en un dataframe
        df = pd.read_json(f)
        #Para cada columna importante
        for col in columnas:
            #Si no está en el archivo
            if col not in df.columns:
                #Se agrega una columna con valores nan
                df[col]=np.nan
        #Se seleccionan unicamente las columnas importantes
        df = df[['prec','tmin','tmax','dir','velmedia','racha', 'sol']]
        #Se sustituyen los valores 88 (indeterminado) por nan en la direccion del viento
        df['dir'] = df['dir'].apply(lambda x: np.nan if x == 88 else x)
        #Se claculan los valores faltantes de cada columna
        nulls = df.isnull().sum()
        #Se suma el total de nulos en cada dataframe
        total = nulls['prec'] + nulls['tmin'] + nulls['tmax'] + nulls['dir'] + nulls['velmedia'] + nulls['racha'] + nulls['sol']
        #Se crea una nuea fila que contiene todos los datos importantes que se quieren mostrar
        row = [filename[2:-5],nulls['prec'],nulls['tmin'],nulls['tmax'],nulls['dir'],nulls['velmedia'],nulls['racha'],nulls['sol'],total]
        #Se añade la fila creada al dataframe resultado
        result.loc[len(result)] = row
        #se cierra el archivo
        f.close()
#Se agrrupan las filas de la misma estación meteorológica        
result_ordered = result.groupby('filename').sum()
#Se ordenan por el total de nulos de menor a mayor
result_ordered = result_ordered.sort_values(by='total')
#Se muestra el resultado
print(result_ordered)