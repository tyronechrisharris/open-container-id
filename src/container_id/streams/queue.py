import threading
from collections import deque
from typing import Any


class LatestFrameQueue:
    """
    A thread-safe bounded queue that drops the oldest pending frame when full.
    Stream freshness is prioritized over processing every frame.
    """

    def __init__(self, maxsize: int = 3):
        self.maxsize = maxsize
        self._queue: deque[Any] = deque(maxlen=maxsize)
        self._condition = threading.Condition()

    def put(self, item: Any) -> None:
        """Add an item to the queue, dropping the oldest if full."""
        with self._condition:
            if len(self._queue) == self.maxsize:
                # deque drops the oldest item implicitly when maxlen is reached
                self._queue.append(item)
            else:
                self._queue.append(item)
            self._condition.notify()

    def get(self, timeout: float | None = None) -> Any | None:
        """Remove and return an item from the queue."""
        with self._condition:
            if not self._queue and not self._condition.wait(timeout=timeout):
                return None
            return self._queue.popleft()

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        with self._condition:
            return len(self._queue)

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        with self._condition:
            return len(self._queue) == 0
