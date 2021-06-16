# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:24:29 2021

@author: unlu
"""

def representacion_documentos(textos_train, textos_test, estrategia, MAX_TKS):
  ''' esta función recibe las consultas de train y test y genera las 
  features dinámicas en base a 
  5 estrategias = {BASELINE, BOW, TFIDF, 3-4-NGRAM-CHAR, 1-2-NGRAM-WORD} 
  '''
  # Vamos a probar 4 estrategias de representación de documentos
  from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  import pandas as pd
  
  # Inicializamos el vectorizer de acuerdo a la estrategia de representación
  if(estrategia=="BASELINE"):
    vectorizer = CountVectorizer(max_features=MAX_TKS)

  # Inicializamos el vectorizer de acuerdo a la estrategia de representación
  if(estrategia=="BOW"):
    vectorizer = CountVectorizer(token_pattern = '[\w\/\%]+', max_features=MAX_TKS)

  elif(estrategia=="BINARIO"):
    vectorizer = CountVectorizer(token_pattern = '[\w\/\%]+', max_features=MAX_TKS, binary=True)

  elif(estrategia=="TFIDF"):
    vectorizer = TfidfVectorizer(token_pattern = '[\w\/\%]+', max_features=MAX_TKS)

  elif(estrategia=="3-4-NGRAM-CHARS"):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,4), token_pattern = '[\w\/\%]+', max_features=MAX_TKS)

  elif(estrategia=="1-2-NGRAM-WORDS"):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), token_pattern = '[\w\/\%]+', max_features=MAX_TKS)

  # Entrenamos el vectorizer para train y test
  representacion_correos_train = vectorizer.fit_transform(textos_train)
  representacion_correos_test = vectorizer.transform(textos_test)     

  # Convertimos las matrices ralas a series para concaterlas con los df originales luego
  df_representacion_correos_train = pd.DataFrame.sparse.from_spmatrix(representacion_correos_train, columns=vectorizer.get_feature_names())
  df_representacion_correos_test = pd.DataFrame.sparse.from_spmatrix(representacion_correos_test, columns=vectorizer.get_feature_names())
  
  return df_representacion_correos_train, df_representacion_correos_test


def gridsearch_por_estrategia_representacion(train, test, estrategia, tecnica, parameters, results_save, atr_consulta='Consulta', atr_clase='clase'):
  from funciones_dataset import consolidar_df
  from sklearn.pipeline import Pipeline
  from sklearn.model_selection import GridSearchCV
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  import pandas as pd

  # Esta función va dentro de un iterador entre las 5 estrategias    
  print('Estrategia de representación: {}' . format(estrategia))
  correos_train_vec, correos_test_vec = representacion_documentos(train[atr_consulta], test[atr_consulta], estrategia, None)

  # Separo en x e y - train y test- (además consolido feature estáticas con dinámicas)
  x_train, y_train = consolidar_df(train, correos_train_vec, atr_consulta, atr_clase)
  x_test, y_test = consolidar_df(test, correos_test_vec, atr_consulta, atr_clase)
  
  # Escalado de datos: Se probó scale y MinMaxScaler y dió mejores resultados el último
  scaler = MinMaxScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.fit_transform(x_test)

  # Dejo Pipeline, si bien no tiene sentido, queda por retrocompatibilidad con versión anterior
  if tecnica == 'SVM':
    from sklearn.svm import SVC
    model_pipe = Pipeline([(tecnica, SVC())])
  elif tecnica == 'LR':
    from sklearn.linear_model import LogisticRegression
    model_pipe = Pipeline([(tecnica, LogisticRegression())])
  else:
    from xgboost import XGBClassifier
    model_pipe = Pipeline([(tecnica, XGBClassifier())])

    
  # Instancio y "entreno" el GridSearchCV
  grid_search=GridSearchCV(model_pipe, param_grid=parameters, cv=None, n_jobs=-1, verbose=3)
  grid_search.fit(x_train_scaled, y_train)

  # Se realizan las predicciones sobre el conjunto de validación
  grid_predictions = grid_search.predict(x_test_scaled) 

  # Paso los resultados y el accuracy a un dataframe
  results_train = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]), pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
  results_train['estrategia'] = estrategia
  
  # Los guardo
  results_train.to_csv('results_train.csv', mode='a')
 
  # Calculo las métricas sobre test para el paper
  acc_test = accuracy_score(y_test, grid_predictions)
  precision_test = precision_score(y_test, grid_predictions, average='macro')
  recall_test = recall_score(y_test, grid_predictions, average='macro')
  f1_test = f1_score(y_test, grid_predictions, average='macro')
 
  # Genero un diccionario con los parámetro y el acc en test
  dict_grid_test = grid_search.best_params_
  dict_grid_test['clasificador'] = tecnica
  dict_grid_test['estrategia'] = estrategia
  dict_grid_test['accuracy'] = acc_test
  dict_grid_test['precision'] = precision_test
  dict_grid_test['recall'] = recall_test
  dict_grid_test['f1_score'] = f1_test

  if results_save=='drive':

    # Paso el diccionario a dataframe  
    results_test = pd.DataFrame([dict_grid_test])

    # Lo guardo en un csv
    results_test.to_csv('results_test.csv', mode='a')
    # Autenticación y guardado en Drive
    import os
    from google.colab import drive
    drive.mount('drive')
    os.system('results_test.csv drive/My Drive/')

  
  print('Estrategia de representación: {}' . format(estrategia))
  print('Parámetros: {}' . format(grid_search.best_params_))
  print('Accuracy Test-Set: {}' . format(acc_test))
  print('Métricas sobre Test-Set: {}' . format(dict_grid_test))

  return grid_search, x_test_scaled, y_test
  
