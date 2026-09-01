from fastapi import FastAPI
import os
#from typing import List, Optional
from src.api.routers import semantic_search, lexical_search, hybrid_search, answer_generation

app = FastAPI(title="REX IPN Search API")

@app.get("/")
async def root():
    return {"message": "Hello, this is the API of use-case 202"}

app.include_router(semantic_search.router_semantic)
app.include_router(lexical_search.router_lexical)
app.include_router(hybrid_search.router_hybrid)
app.include_router(answer_generation.router_LLM)


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
        # For production, we use multiple worker processes to handle requests in parallel.
        # The number of workers can be set via the WEB_CONCURRENCY environment variable,
        # or it defaults to a recommended value based on CPU cores.
        default_workers = (os.cpu_count() or 1) * 2 + 1
        num_workers = int(os.environ.get("WEB_CONCURRENCY", default_workers))
        uvicorn.run(app, host=host, port=port, workers=num_workers)