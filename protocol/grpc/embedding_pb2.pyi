from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EncodeRequest(_message.Message):
    __slots__ = ("sentences",)
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    sentences: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, sentences: _Optional[_Iterable[str]] = ...) -> None: ...

class EncodeResponse(_message.Message):
    __slots__ = ("embeddings", "embedding_dim")
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    embeddings: _containers.RepeatedScalarFieldContainer[float]
    embedding_dim: int
    def __init__(self, embeddings: _Optional[_Iterable[float]] = ..., embedding_dim: _Optional[int] = ...) -> None: ...

class CreateSimilarityRequest(_message.Message):
    __slots__ = ("model", "source_sentence", "sentences")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SENTENCE_FIELD_NUMBER: _ClassVar[int]
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    model: str
    source_sentence: str
    sentences: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model: _Optional[str] = ..., source_sentence: _Optional[str] = ..., sentences: _Optional[_Iterable[str]] = ...) -> None: ...

class SimilarityResult(_message.Message):
    __slots__ = ("sentence", "similarity")
    SENTENCE_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    sentence: str
    similarity: float
    def __init__(self, sentence: _Optional[str] = ..., similarity: _Optional[float] = ...) -> None: ...

class CreateSimilarityResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SimilarityResult]
    def __init__(self, results: _Optional[_Iterable[_Union[SimilarityResult, _Mapping]]] = ...) -> None: ...
