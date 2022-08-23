# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:10:56 2021

@author: Juan Manuel Fernandez
"""
import time
from funciones_IEEE import cargar_dataset, get_metricas, get_BoW
from funciones_preprocesamiento import preprocesar_correos
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def main():
    return 1

if __name__ == '__main__':
    col_numericas = ['dia_semana', 'semana_del_mes', 'mes', 'cuatrimestre', 'anio', 'hora_discretizada', 'dni_discretizado', 'legajo_discretizado', 'posee_legajo', 'posee_telefono', 'carrera_valor', 'proveedor_correo', 'cantidad_caracteres', 'proporcion_mayusculas', 'proporcion_letras', 'cantidad_tildes', 'cantidad_palabras', 'cantidad_palabras_cortas', 'proporcion_palabras_distintas', 'frecuencia_signos_puntuacion', 'cantidad_oraciones', 'utiliza_codigo_asignatura']
    DIR_data = 'https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/50jaiio/consolidados/'
    DIR_data = 'data/'
    MAX_TKS = None
    BOOSTING = '-boosting'
    ATR_ESTATICOS = True
    INSTANCIAS_MANUALES = True
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
    
    filtro_atributo_X = ['consulta']
    if ATR_ESTATICOS:
        filtro_atributo_X = False
    
    print('Cargo las consultas etiquetadas de forma manual para TRAIN...')
    X_manual_train, y_manual_train = cargar_dataset(DIR_data, 'correos-train-jaiio-80.csv', descarga=False, atr_X=filtro_atributo_X, atr_y='clase')    
    X_manual_train['consulta'] = preprocesar_correos(X_manual_train['consulta'])

    print('\nCargo los datos etiquetados de forma manual para TEST...')
    X_manual_test, y_manual_test = cargar_dataset(DIR_data, 'correos-test-jaiio-20.csv', descarga=False, atr_X=filtro_atributo_X, atr_y='clase')
    X_manual_test['consulta'] = preprocesar_correos(X_manual_test['consulta'].to_list())

    # Escalo las variables numéricas
    scaler = MinMaxScaler()
    X_manual_train[col_numericas] = scaler.fit_transform(X_manual_train[col_numericas])
    X_manual_test[col_numericas] = scaler.fit_transform(X_manual_test[col_numericas])

    # Realizo la búsqueda grid para los datos etiquetados de forma manual
    # Itero en 4 estrategias de representación de documentos definidas para get_BoW()
    for estrategia_representacion in estrategias_representacion:
        
        # Calculo la hora actual
        hora_actual = time.strftime('%H:%M:%S', time.localtime())
        print(f'\n({hora_actual}) Etiquetado manual-{estrategia_representacion}: ', sep="")

        # Genero la bolsa de palabras con la variante que toca (train y test)
        bag_of_words, X_bw_test = get_BoW(X_manual_train['consulta'], estrategia_representacion, X_test=X_manual_test['consulta'])
        
        # Inicializo el modelo manual
        grid = GridSearchCV(SVC(), params_svm, cv=5)

        # Si incorporo los estaticos
        if ATR_ESTATICOS:
            # Concateno los estáticos a los dinámicos
            X_train = pd.concat([X_manual_train.drop(['consulta'], axis=1), bag_of_words], axis=1)
            X_test = pd.concat([X_manual_test.drop(['consulta'], axis=1), X_bw_test], axis=1)

            # Escalado de datos: Se probó scale y MinMaxScaler y dió mejores resultados el último
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.fit_transform(X_test)
    
            grid.fit(X_train_scaled, y_manual_train)
    
            # Genero las clases para los casos de prueba, según el modelo
            predictions = grid.predict(X_test_scaled)
        else: # Si solo tomo la consulta entreno sólo en función del bag of words
            grid.fit(bag_of_words.values, y_manual_train)
    
            # Genero las clases para los casos de prueba, según el modelo
            predictions = grid.predict(X_bw_test.values)
            
        # Calculo las métricas según Y_real y Y_test
        df_metricas = get_metricas(y_manual_test,
                                   predictions,
                                   'SVM',
                                   'Etiquetado manual',
                                   estrategia_representacion,
                                   str(grid.get_params()),
                                   df_metricas)
        
        # Calculo la hora actual
        hora_actual = time.strftime('%H:%M:%S', time.localtime())
        
        # Tomo el accuracy en test
        acc_test = float(df_metricas.iloc[-1:]['accuracy'])

        # Imprimo los datos
        print(f'({hora_actual}) Accuracy en test: {acc_test}.', sep="")
        
        # Vemos un reporte de clasificación de varias métricas
        from sklearn import metrics #Importar el módulo metrics de scikit-learn
        print(metrics.classification_report(y_manual_test, predictions))


    # Persisto los resultados en un xlsx
    df_metricas.to_excel('metricas-v2-1-fe.xlsx')