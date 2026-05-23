from __future__ import annotations

from app.models import ChatMessage


class ConversationStore:
    def __init__(self) -> None:
        self._store: dict[str, list[ChatMessage]] = {}

    def save(self, response_id: str, messages: list[ChatMessage]) -> None:
        self._store[response_id] = messages

    def get(self, response_id: str) -> list[ChatMessage] | None:
        return self._store.get(response_id)

    def delete(self, response_id: str) -> None:
        self._store.pop(response_id, None)

    def clear(self) -> None:
        self._store.clear()


store = ConversationStore()