from fastapi import APIRouter
from src.api.base_models.base_model import SearchRequest, RetrievalResult
from fastapi.concurrency import run_in_threadpool
from src.api.constants.api_launch import API_ROUTER_PREFIX
from pathlib import Path
import sys

sys.path.append(str(Path().resolve().parent))
from src.api.utils.query import combined_search


router_hybrid = APIRouter(
    tags=["hybrid_search"],
)


@router_hybrid.post("/hybrid")
async def search_query(query: SearchRequest, response_model=list[RetrievalResult]):
    try:
        # Run the blocking search function in a thread pool to not block the event loop
        result = await run_in_threadpool(
            combined_search, query.query, query.top_k, query.use_dictionary, query.filters
        )

        return {
            "chunks": result[f"rrf_top {query.top_k}"],
            "timings": result["timings"]
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}
