# -*- coding: utf-8 -*-

# Cargo las librerias
import pandas as pd
import numpy as np

def terms2df(conx, index, n_results, query):

    response = conx.search(
        index=index,
        body=query,
        size=n_results)

    elastic_docs = response["hits"]["hits"]
    
    fields = {}
    score_list = []
    for num, doc in enumerate(elastic_docs):
        source_data = doc["_source"]
        score_list.append(doc["_score"])
        for key, val in source_data.items():
            try:
                fields[key] = np.append(fields[key], val)
            except KeyError:
                fields[key] = np.array([val])

    results_df = pd.DataFrame(fields)
    results_df['score'] = pd.Series(score_list)
    
  
    return results_df


def terms2df_tfidf_ss3(conx, index, query_terms, n_results):

    query = {
          "query": {
            "match": {
              "consulta": {
                "query": query_terms,
                "operator": "or",
                 "fuzziness": "AUTO",
                 "auto_generate_synonyms_phrase_query" : "true"
              }
            }
           }
          }

    results_df = terms2df(conx, index, n_results, query)
    
    return results_df


def terms2df_lr(conx, index, query_terms_pos, query_terms_neg, n_results):

    query = {
      "query": {
        "boosting": {
          "positive": {
            "match": {
              "consulta": query_terms_pos
            }
          },
          "negative": {
            "match": {
              "consulta": query_terms_neg
            }
          },
          "negative_boost": 0.1
        }
      }
    }

    results_df = terms2df(conx, index, n_results, query)
    
    return results_df


def limpiar_clase_ss3(archivo):
    """
    Esta funcion recibe el nombre del archivo generado por SS3
    y limpia el nombre de la clase
    """
    # ss3_vocab_Boleto Universitario(words).csv
    aux = archivo.split('(')[0]
    clase = aux.split('_')[2]
    
    return clase
