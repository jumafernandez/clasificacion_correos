# Cargo las librerías
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np

def terms2df_tfidf(conx, index, query_terms, n_results):

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
