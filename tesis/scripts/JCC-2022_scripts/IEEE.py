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
import pandas as pd

def main():
    return 1

if __name__ == '__main__':
        
    DIR_data = 'https://raw.githubusercontent.com/jumafernandez/clasificacion_correos/main/data/50jaiio/consolidados/'
    DIR_data = 'data/'
    MAX_TKS = None
    BOOSTING = '-boosting'
    INSTANCIAS_MANUALES = False
    df_metricas = pd.DataFrame()

    # Defino las estrategias de representación de documentos
    estrategias_representacion = ['3-4-NGRAM-CHARS']

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

    # Realizo la búsqueda grid para los datos etiquetados de forma manual
    # Itero en 4 estrategias de representación de documentos definidas para get_BoW()
    for estrategia_representacion in estrategias_representacion:
        
        # Calculo la hora actual
        hora_actual = time.strftime('%H:%M:%S', time.localtime())
        print(f'\n({hora_actual}) Etiquetado manual-{estrategia_representacion}: ', sep="")

        # Genero la bolsa de palabras con la variante que toca (train y test)
        bag_of_words, X_test = get_BoW(X_manual_train, estrategia_representacion, X_test=X_manual_test)
    
        # Entreno el modelo manual
        grid = GridSearchCV(SVC(), params_svm, cv=5)
        grid.fit(bag_of_words, y_manual_train)
    
        # Genero las clases para los casos de prueba, según el modelo
        predictions = grid.predict(X_test)
        
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

    # COMIENZAN LOS SISTEMAS DE VOTACIÓN (Manual + 1 estrategia)
    # Defino las estrategias de feature extraction (FE) para iterar en la búsqueda grid
    estrategias = ['lr', 'tfidf', 'ss3']
    
    for estrategia_fe_1 in estrategias:

        # Calculo la hora actual
        hora_actual = time.strftime('%H:%M:%S', time.localtime())

        # En el primer bucle corro la primera estrategia únicamente
        print(f'\n({hora_actual}) Corro la estrategia {estrategia_fe_1}.')

        # Cargo el dataset para la estrategia de FE y preproceso las consultas
        URL_data_1 = f'dataset-{estrategia_fe_1}-200{BOOSTING}-prep.csv'
        X_1, y_1 = cargar_dataset(DIR_data, URL_data_1, descarga=False, atr_X=['consulta'], atr_y='clase', CANTIDAD_INSTANCIAS=100)
        X_1 = preprocesar_correos(X_1['consulta'].to_list())
    
        # Incorporo los ejemplos de la estrategia a los etiquetados manualmente
        if INSTANCIAS_MANUALES:
            X_train_fe_1 = X_manual_train + X_1
            y_train = y_manual_train + y_1
        else:
            X_train_fe_1 = X_1
            y_train = y_1

        # Itero en 4 estrategias de representación de documentos definidas para get_BoW()
        estrategias_representacion = ['3-4-NGRAM-CHARS', '1-2-NGRAM-WORDS']
        for estrategia_representacion in estrategias_representacion:

            # Calculo la hora actual
            hora_actual = time.strftime('%H:%M:%S', time.localtime())

            print(f'\n({hora_actual}) {estrategia_fe_1}-{estrategia_representacion}: ', sep="")
            
            # Genero la bolsa de palabras con la variante que toca (train)
            X_train, X_test = get_BoW(X_train_fe_1, estrategia_representacion, X_test=X_manual_test)
        
            # Instancio la búsqueda grid con los parámetros
            grid = GridSearchCV(SVC(), params_svm, cv=5)
            grid.fit(X_train, y_train)
        
            # Realizo las predicciones
            predictions = grid.predict(X_test)
        
            # Calculo las métricas
            df_metricas = get_metricas(y_manual_test, 
                                       predictions, 
                                       'SVM', 
                                       estrategia_fe_1,
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