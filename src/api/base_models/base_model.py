
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import List, Literal, Optional


class SearchRequest(BaseModel):
    query: str = Field(
        default="Dans le cadre du projet EPR2 et des estimations de volumes des produits dangereux et autres ICPE",
        description="Query text",
    )

    top_k: int = Field(5, description="Number of results to return")
    use_dictionary: bool = Field(False, description="Whether to use the dictionary and replace the bigrams/trigrams/multigrams etc with their different definitions or not")
    filters: Optional[dict] = None

class RetrievalResult(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier of the chunk")
    chunk_content: str = Field(..., description="Text content of the retrieved chunk")
    original_doc_id: str = Field(
        ..., description="Document ID from which the chunk was extracted"
    )
    score: float = Field(..., description="Similarity score returned by search engine")
    source: Literal["semantic", "lexical", "hybrid"] = Field(
        ..., description="Type of search that retrieved this chunk"
    )
@dataclass
class LLMInput:
    query: str
    chunks: List[RetrievalResult]