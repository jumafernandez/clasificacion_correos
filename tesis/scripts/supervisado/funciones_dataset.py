# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 22:46:55 2021

@author: unlu
"""

def get_clases_pre():
  '''
  Esta función retorna las etiquetas de las clases sobre el total de los correos. 
  Tomado de notebooks/jcc/de train_test_data.ipynb
  Utilizada antes de unificar clases
  '''
  import numpy as np

  etiquetas_clases = np.array(['Boleto Universitario', 
                               'Cambio de Carrera', 
                               'Cambio de Comisión',
                               'Carga de Notas', 
                               'Certificados Web', 
                               'Consulta por Equivalencias',
                               'Consulta por Legajo', 
                               'Consulta sobre Título Universitario',
                               'Cursadas', 
                               'Datos Personales', 
                               'Exámenes',
                               'Ingreso a la Universidad',
                               'Inscripción a Cursadas',
                               'Pedido de Certificados',
                               'Problemas con la Clave',
                               'Reincorporación', 
                               'Requisitos de Ingreso',
                               'Simultaneidad de Carreras', 
                               'Situación Académica',
                               'Vacunas Enfermería'])

  return etiquetas_clases


def get_clases():
  '''
  Esta función retorna las etiquetas de las clases sobre el total de los correos. 
  Tomado de notebooks/jcc/de train_test_data.ipynb
  '''
  import numpy as np

  etiquetas_clases = np.array(['Boleto Universitario', 
                               'Cambio de Carrera', 
                               'Cambio de Comisión',
                               'Carga de Notas', 
                               'Certificados Web', 
                               'Consulta por Equivalencias',
                               'Consulta por Legajo', 
                               'Consulta sobre Título Universitario',
                               'Cursadas', 
                               'Datos Personales', 
                               'Exámenes',
                               'Ingreso a la Universidad',
                               'Inscripción a Cursadas',
                               'Pedido de Certificados',
                               'Problemas con la Clave',
                               'Reincorporación', 
                               'Requisitos de Ingreso',
                               'Simultaneidad de Carreras', 
                               'Situación Académica',
                               'Vacunas Enfermería'])

  return etiquetas_clases

def cargar_dataset(URL_data, file_train, file_test, descarga, nombre_clase, class_labels, cantidad_clases, texto_otras):
  '''
  Carga los train y test set y genera la reducción de clases, en caso que sea necesario
  '''
  import pandas as pd
  import numpy as np
  import warnings
  warnings.filterwarnings("ignore")

  # Genero el enlace completo
  URL_file_train = URL_data + file_train
  URL_file_test = URL_data + file_test
  
  if descarga:
    print('Se inicia descarga de los datasets.'.format(file_train, file_test))
    import wget
    wget.download(URL_file_train)
    wget.download(URL_file_test)
  else:
    file_train = URL_file_train
    file_test = URL_file_test
    
  # Leemos el archivo en un dataframe
  df_train = pd.read_csv(file_train)
  df_test = pd.read_csv(file_test)

  # Agrupamiento de clases
  etiquetas_clases = class_labels
  if cantidad_clases<len(class_labels):
    # Se realiza un conteo de frecuencia por clase y se toman los correos que pertenecen a 
    # las N-cantidad_clases menos observadas
    clases = df_train.clase.value_counts()
    clases_minoritarias = clases.iloc[cantidad_clases-1:].keys().to_list()

    # Agrego a las etiquetas la etiqueta "Otras Consultas" para el agrupamiento
    etiquetas_clases = np.append(class_labels, texto_otras)

    # Genero una nueva clave de clases para "Otras Consultas" a modo de agrupar las que poseen menos apariciones
    df_train.clase[df_train[nombre_clase].isin(clases_minoritarias)] = np.where(etiquetas_clases == texto_otras)[0]
    df_test.clase[df_test[nombre_clase].isin(clases_minoritarias)] = np.where(etiquetas_clases == texto_otras)[0]

  print("\nEl conjunto de entrenamiento tiene la dimensión: " + str(df_train.shape))
  print("El conjunto de testeo tiene la dimensión: " + str(df_test.shape))

  return df_train, df_test, etiquetas_clases


def consolidar_df(df, features_dinamicas_vec, atributo_consulta, atributo_clase):
  '''
  Función para unir features dinámicas a las estáticas y separar en x e y
  '''
  import pandas as pd
  
  # Separo en x e y -train-
  y = df[atributo_clase].to_numpy()
  x = pd.concat([df.drop([atributo_consulta, atributo_clase], axis=1), features_dinamicas_vec], axis=1)
  
  return x, y

  
def add_clase(array, clase):
  '''
  Creo esta función para agregar la etiqueta Otras consultas
  '''
  import numpy as np
  array_nuevo = np.append(array, clase)
  return array_nuevo

def separar_x_y_rna(df, atributo_consulta, atributo_clase):
  '''
  Función para separar en x e y
  '''
  # Separo en x e y
  X = df[atributo_consulta]
  y = df[atributo_clase].to_numpy()

  return X, y  
