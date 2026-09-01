from fastapi import  APIRouter
from src.api.base_models.base_model import SearchRequest, RetrievalResult
from src.api.constants.api_launch import API_ROUTER_PREFIX
from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent))
from src.api.utils.query import combined_search



router_lexical = APIRouter(
    prefix=f"{API_ROUTER_PREFIX}/lexical_search",
    tags=["lexical_search"],
)


@router_lexical.post("/lexical")
async def search_query(query: SearchRequest, response_model=list[RetrievalResult]):
    try:
        result = combined_search(query.query, query.top_k, query.use_dictionary)
        return result[f"lexical_top {query.top_k}"]
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}
