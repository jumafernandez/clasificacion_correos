# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:20:36 2022

@author: Juan
"""

import time
from funciones_IEEE import cargar_dataset, get_metricas, get_BoW
from funciones_preprocesamiento import preprocesar_correos
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd

DIR_data = 'https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/50jaiio/consolidados/'
DIR_data = 'data/'
MAX_TKS = None
INSTANCIAS_MANUALES = False
df_metricas = pd.DataFrame()

# Defino las estrategias de representación de documentos
estrategias_representacion = ['BINARIO', 'TFIDF', '3-4-NGRAM-CHARS', '1-2-NGRAM-WORDS']

# Defino los parámetros para GridSearchCV
params_svm = {'C': [0.1, 1, 10], 
              'gamma': [0.01, 0.1, 1],
              'class_weight': [None, 'balanced'],
              'kernel': ['rbf', 'linear', 'sigmoid'],
              'probability': [False]
              }

print('Cargo las consultas etiquetadas de forma manual para TRAIN...')
X_manual_train, y_manual_train = cargar_dataset(DIR_data, 'correos-train-jaiio-80.csv', descarga=False, atr_X=['consulta'], atr_y='clase')
X_manual_train = preprocesar_correos(X_manual_train['consulta'].to_list())

print('\nCargo los datos etiquetados de forma manual para TEST...')
X_manual_test, y_manual_test = cargar_dataset(DIR_data, 'correos-test-jaiio-20.csv', descarga=False, atr_X=['consulta'], atr_y='clase')
X_manual_test = preprocesar_correos(X_manual_test['consulta'].to_list())

# Cargo el dataset para lr
print('Cargo las consultas etiquetadas por lr...')
URL_data_1 = f'dataset-lr-200-prep.csv'
X_1, y_1 = cargar_dataset(DIR_data, URL_data_1, descarga=False, atr_X=['consulta'], atr_y='clase')
X_1 = preprocesar_correos(X_1['consulta'].to_list())

# Cargo el dataset para ss3
print('Cargo las consultas etiquetadas por ss3...')
URL_data_2 = f'dataset-ss3-200-prep.csv'
X_2, y_2 = cargar_dataset(DIR_data, URL_data_2, descarga=False, atr_X=['consulta'], atr_y='clase')
X_2 = preprocesar_correos(X_2['consulta'].to_list())

# Cargo el dataset para tfidf
print('Cargo las consultas etiquetadas por tfidf...')
URL_data_3 = f'dataset-tfidf-200-prep.csv'
X_3, y_3 = cargar_dataset(DIR_data, URL_data_3, descarga=False, atr_X=['consulta'], atr_y='clase')
X_3 = preprocesar_correos(X_3['consulta'].to_list())

# Calculo la hora actual
hora_actual = time.strftime('%H:%M:%S', time.localtime())
                
# En el primer bucle corro la primera estrategia únicamente
print(f'\n({hora_actual}) lr-ss3-tfidf.')
                
X_train_join = []
y_train_join = []
                    
# Genero 2 listas con las coincidencias entre las estrategias
for i in range(len(X_1)):
    
    # Se verifica si coinciden las dos primeras estrategias
    if (X_1[i] in X_2) and (y_1[i] == y_2[X_2.index(X_1[i])]):

        # Se verifica si también coinciden con la tercera
        if (X_1[i] in X_3) and (y_1[i] == y_3[X_3.index(X_1[i])]):

            #print(f'{i}-{X_1[i]}')
            X_train_join.append(X_1[i])
            y_train_join.append(y_1[i])
                            
print(f'Coinciden {len(X_train_join)} instancias entre lr-ss3-tfidf.')

# Incorporo los ejemplos de la estrategia a los etiquetados manualmente
if INSTANCIAS_MANUALES:
    X_train_fe_2 = X_manual_train + X_train_join
    y_train = y_manual_train + y_train_join
else:
    X_train_fe_2 = X_train_join
    y_train = y_train_join



for estrategia_representacion in estrategias_representacion:

    print(f'\n({hora_actual}) lr-ss3-tfidf, {estrategia_representacion}: ', sep="")                   
            
    # Genero la bolsa de palabras con la variante que toca (train y test)
    bag_of_words, X_test = get_BoW(X_train_fe_2, estrategia_representacion, X_test=X_manual_test)
                            
    # Entreno el modelo manual
    grid = GridSearchCV(SVC(), params_svm, cv=5)
    grid.fit(bag_of_words, y_train)
    
    # Genero las clases para los casos de prueba, según el modelo
    predictions = grid.predict(X_test)
        
    # Calculo las métricas según Y_real y Y_test
    df_metricas = get_metricas(y_manual_test,
                           predictions,
                           'SVM',
                           'lr-ss3-tfidf',
                           estrategia_representacion,
                           str(grid.get_params()),
                           df_metricas)
        
    # Calculo la hora actual
    hora_actual = time.strftime('%H:%M:%S', time.localtime())
        
    # Tomo el accuracy en test
    acc_test = float(df_metricas.iloc[-1:]['accuracy'])
    
    # Imprimo los datos
    print(f'({hora_actual}) Accuracy en test: {acc_test}.\n')
    
# Persisto los resultados en un xlsx
df_metricas.to_excel('metricas-v2-3-fe.xlsx')