# Predicción de los niveles de polen en Madrid

En este repositorio se encuentra la implementación de 4 modelos de predicción de polen y una interfaz web a través de la cual poder visualizar sus resultados.
Este proyecto ha sido realizado por Iulius Gherasim, Jun Qiu y Jaime Pastrana García para la asignatura de DASI (Desarrollo de Aplicaciones y Servicios Inteligentes) de la UCM.


# Instrucciones de ejecución
## 1. Instalación de dependencias

En el archivo `requirements.txt` se pueden encontrar los requerimientos del proyecto en cuanto a librerías de python se refiere. Para asegurar la correcta instalación de todas las librerías, se sugiere emplear el siguiente comando:

    pip install -r requirements.txt


## 2. Preprocesado y análisis de los datos y entrenamiento de los modelos

Las carpetas se encuentran numeradas en el orden en que deben ejecutarse.
Además, toda la implementación relacionada con el tratamiento de los datos y el entrenamiento de los modelos se ha realizado sobre cuadernos Jupyter Notebook.
Como consecuencia, la ejecución de esta fase del proyecto consiste en ejecutar los cuadernos Jupyter Notebook situados en las carpetas 1, 2 y 3 en ese orden.
En estos mismos cuadernos se pueden encontrar anotaciones en formato markdown que explican los distintos procesamientos realizados.

## 3. Despliegue del modelo en una interfaz web

Para poder emplear el modelo en un contexto de uso real, se ha creado una interfaz web que permite obtener predicciones en tiempo real mediante los modelos entrenados.
Para su ejecución es necesario abrir una terminal en la carpeta `4 Interfaz` y ejecutar el siguiente comando:

    python3 app.py

Tras ello, basta con abrir la página web en el enlace local http://127.0.0.1:5000/.

Por último, es importante destacar que los modelos no podrán realizar predicciones adecuadas si no hay datos actualizados en las APIs de las que se alimenta.
Como, desgraciadamente, esto es muy común, se ha dejado la posibilidad de descomentar la línea de código 23 del archivo `app.py` y comentar la línea 24, modificándose así el día de la predicción a uno en el que sí que hay datos.
