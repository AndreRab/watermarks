from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar


MediaT = TypeVar("MediaT")
PayloadT = TypeVar("PayloadT")
GenerationRequestT = TypeVar("GenerationRequestT")


@dataclass
class DetectionResult(Generic[PayloadT]):
    detected: bool
    payload: PayloadT | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class WaterMarkTransformer(ABC, Generic[MediaT, PayloadT, GenerationRequestT]):
    can_embed: ClassVar[bool] = False
    can_generate: ClassVar[bool] = False
    can_recover: ClassVar[bool] = False

    def embed(self, media: MediaT, payload: PayloadT) -> MediaT:
        pass

    def generate(self, request: GenerationRequestT) -> MediaT:
        pass

    @abstractmethod
    def detect(self, media: MediaT) -> DetectionResult[PayloadT]:
        pass

    def recover_original(self, media: MediaT) -> MediaT:
        pass