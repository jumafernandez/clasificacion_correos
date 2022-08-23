# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 08:13:02 2021

@author: Juan
"""

def cargar_dataset(URL_data, file, descarga, atr_X, atr_y, CANTIDAD_INSTANCIAS=None):
    '''
    Carga un dataset en:
        X: dataframe con los atributos
        y: lista con las clases

    Parámetros:
        URL_data: directorio de los archivos (o URL del directorio)
        file: nombre del archivo
        atr_X: atributos para el vector X (en nuestro caso sólo consulta)
        atr_y: atributos para el target y
        CANTIDAD_INSTANCIAS: Por si limita la cantidad de instancias (feature extraction)
    '''
    import pandas as pd
    import wget
    import warnings
    warnings.filterwarnings("ignore")
           
    # Genero el enlace completo
    URL_file = URL_data + file
      
    if descarga:
        print('Se inicia descarga de los datasets.')
        wget.download(URL_file)
    else:
        file = URL_file
        
    # Leemos el archivo en un dataframe
    df = pd.read_csv(file)

    if CANTIDAD_INSTANCIAS:
        df = df.sort_values(['clase','score'], ascending=False).groupby('clase').head(CANTIDAD_INSTANCIAS).reset_index(drop=True)
        df = df.sample(frac = 1)

    print(f"El dataset tiene la dimensión: {df.shape}.")

    X = df[atr_X]
    y = df[atr_y].tolist()

    return X, y

def get_metricas(y_true, y_pred, clasificador, estrategia, representacion, parametros, df=None):
    """
    Devuelve un dataframe con las métricas de la corrida.
    Si se le pasa un dataframe incorpora la fila, sino crea uno nuevo

    Parameters
    ----------
    y_true : list
        Es el vector con las clases reales\n
    y_pred : list
        Es el vector con las clases generadas por el modelo\n
    clasificador : string
        Es una cadena de caracteres con el descriptor del clasificador\n
    estrategia: string
        Es una cadena de caracteres con la estrategia de feature extraction\n
    representacion: string
        Es una cadena de caracteres con la estrategia de representación del texto\n
    parametros : string
        Es una cadena de caracteres con los parámetros del clasificador\n
    df : dataframe
        Es el dataframe donde se incorporan las métricas. Si es None se crea uno.\n

    Returns
    -------
    Un dataframe con las métricas

    """
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    import pandas as pd
    
 
    # Calculo las métricas sobre test y genero un diccionario
    metricas = {}
    metricas['clasificador'] = clasificador
    metricas['estrategia_feature_extraction'] = estrategia
    metricas['estrategia_representacion'] = representacion
    metricas['parametros'] = parametros
    metricas['accuracy'] = accuracy_score(y_true, y_pred)
    metricas['precision'] = precision_score(y_true, y_pred, average='weighted')
    metricas['recall'] = recall_score(y_true, y_pred, average='weighted')
    metricas['f1_score'] = f1_score(y_true, y_pred, average='weighted')


    if df is not None:
        df = df.append(metricas, ignore_index=True)
    else:
        df = pd.DataFrame([metricas])    
    
    return df

def get_BoW(X, representacion, MAX_TKS=None, X_test=None):
    """
    Esta función recibe una lista de consultas (o dos) y la devuelve vectorizada
    según la estrategia de representación definida.
    
    Parameters
    ----------
    X : list
        Lista con Consultas para vectorizar\n
    representacion : str
        Es la estrategia de representación de las consultas\n
    MAX_TKS : numeric
        Es la cantidad máxima de features que genera la BoW (opcional)\n
    X_test : list
        Consultas para vectorizar. Se asume que si se reciben dos listas, 
        la segunda es para test (opcional) y X para train\n
    

    Returns
    -------
    Retorna las consultas vectorizadas según una estrategia de representación.

    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    # Tipo de vectorización: 3-4-NGRAM-CHARS   
    if representacion == '3-4-NGRAM-CHARS':
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,4), token_pattern = '[\w\/\%]+', max_features=MAX_TKS).fit(X)
    
    elif(representacion == "BINARIO"):
        vectorizer = CountVectorizer(token_pattern = '[\w\/\%]+', max_features=MAX_TKS, binary=True).fit(X)

    elif(representacion == "TFIDF"):
        vectorizer = TfidfVectorizer(token_pattern = '[\w\/\%]+', max_features=MAX_TKS).fit(X)
        
    elif(representacion == "1-2-NGRAM-WORDS"):
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), token_pattern = '[\w\/\%]+', max_features=MAX_TKS).fit(X)

    X_vec = vectorizer.transform(X)

    if X_test is not None:
        X_test_vec = vectorizer.transform(X_test)

    if X_test is not None:
        return X_vec, X_test_vec
    else:
        return X_vec