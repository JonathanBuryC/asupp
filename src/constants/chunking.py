#MODEL_NAME = "intfloat/multilingual-e5-large"
MODEL_NAME= "/opt/app-root/src/uc202-ipn-rex/src/models/models/test"
TOKENS_PER_CHUNK = 512
OVERLAP = 50
BULK_SIZE=1000
SCROLL_CHUNKING="15m"
SCROLL_PROCESSED_ID="5m"
TEXT_ATTRIBUTE="content"  #nom de l'attribut dans l'index elastic à chunker (c juste la valeur par defaut, sinon tu peux 
#mettre nimporte quel nom dans la methode du chunking)
GET_ALL_QUERY = {"query": {"match_all": {}}}
