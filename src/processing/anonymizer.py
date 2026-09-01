from kitten import Desidentifier
from src.constants.constants import (
    DEFAULT_ANONYMIZER_MODEL,
    DEFAULT_ENTITIES_TO_DEACTIVATE,
)
from typing import Optional, Tuple, List
from dataclasses import dataclass
from src.processing.document import Document

@dataclass
class AnonymizationSegment:
    text: str
    tag: str
    start: int
    end: int
    score: float


class Anonymizer:
    def __init__(self, model_path: str = DEFAULT_ANONYMIZER_MODEL):
        self.desidentifier = Desidentifier.load(model_path)
        self.desidentifier.deactivate(*DEFAULT_ENTITIES_TO_DEACTIVATE)

    def activate_all(self):
        self.desidentifier.activate(*self.desidentifier.available_entities)

    def deactivate_entities(self, *entities: str):
        self.desidentifier.deactivate(*entities)

    def anonymize_text(
        self, text: str, inspection: bool = False
    ) -> Tuple[str, Optional[List[AnonymizationSegment]]]:
        if inspection:
            anonymized_text, segments_raw = self.desidentifier.desidentify(
                text, inspection=True
            )

            segments = [AnonymizationSegment(**vars(seg)) for seg in segments_raw]
            return anonymized_text, segments
        else:
            return self.desidentifier.desidentify(text), None

    def anonymize_document(self, doc:Document) -> Optional[List[AnonymizationSegment]]:
        if not doc.text:
            return None
        anonymized_text, segments = self.anonymize_text(doc.text, inspection=True)
        doc.text = anonymized_text
        return segments
