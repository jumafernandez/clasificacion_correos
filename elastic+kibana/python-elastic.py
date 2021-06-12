from elasticsearch import Elasticsearch

elastic_client = Elasticsearch(hosts=["localhost"])

result = elastic_client.search(
    index="correos",
    body = {
            "query": {
                "match_all": {}
                }
            }
    )
