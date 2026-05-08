from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class MemRequest(_message.Message):
    __slots__ = ("child_id", "child_name", "agent_id", "query", "intent", "limit")
    CHILD_ID_FIELD_NUMBER: _ClassVar[int]
    CHILD_NAME_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    INTENT_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    child_id: str
    child_name: str
    agent_id: str
    query: str
    intent: str
    limit: int
    def __init__(self, child_id: _Optional[str] = ..., child_name: _Optional[str] = ..., agent_id: _Optional[str] = ..., query: _Optional[str] = ..., intent: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class MemResponse(_message.Message):
    __slots__ = ("code", "message", "data")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    code: int
    message: str
    data: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, code: _Optional[int] = ..., message: _Optional[str] = ..., data: _Optional[_Iterable[str]] = ...) -> None: ...
