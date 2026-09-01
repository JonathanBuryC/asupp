from fastapi import FastAPI, Query, APIRouter
from src.api.constants.api_launch import API_ROUTER_PREFIX
from pydantic import BaseModel
import os

# from typing import List, Optional
from notebooks.query import combined_search, semantic_search, lexical_search

app = FastAPI(title="REX IPN Search API")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    use_dictionary: bool = False


@app.get("/")
async def root():
    return {"message": "Hello, this is the API of use-case 202"}


router_semantic = APIRouter(
    prefix=f"{API_ROUTER_PREFIX}/semantic_search",
    tags=["semantic_search"],
)


@router_semantic.post("/semantic")
async def search_query(query: SearchRequest):
    try:
        result = combined_search(query.query, query.top_k, query.use_dictionary)
        return result
        # return result[f"semantic_top {query.top_k}"]
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


router_lexical = APIRouter(
    prefix=f"{API_ROUTER_PREFIX}/lexical_search",
    tags=["lexical_search"],
)


@router_lexical.post("/lexical")
async def search_query_2(query: SearchRequest):
    try:
        result = combined_search(query.query, query.top_k, query.use_dictionary)
        return result[f"lexical_top {query.top_k}"]
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


router_hybrid = APIRouter(
    prefix=f"{API_ROUTER_PREFIX}/hybrid_search",
    tags=["hybrid_search"],
)


@router_hybrid.post("/hybrid")
async def search_query_hybrid(query: SearchRequest):
    try:
        result = combined_search(query.query, query.top_k, query.use_dictionary)
        return result[f"rrf_top {query.top_k}"]
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


@app.get("/test")
async def test(a, b):
    return a + b


app.include_router(router_semantic)
app.include_router(router_lexical)
app.include_router(router_hybrid)
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False") in ["true", "True", "1", "yes", "Yes"]

    if debug:
        import sys

        sys.argv = [
            "uvicorn",
            "src.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload",
        ]
        uvicorn.main()
    else:
        uvicorn.run(app, host=host, port=port)
