from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


T = TypeVar("T")


class InferenceQueueFull(RuntimeError):
    """Raised when the bounded inference queue cannot accept another job."""


class InferenceCoordinatorClosed(RuntimeError):
    """Raised when work is submitted after the coordinator has stopped."""


@dataclass(frozen=True)
class InferenceQueueStatus:
    capacity: int
    queued: int
    active: bool
    submitted: int
    completed: int
    failed: int
    worker_count: int
    closed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "queued": self.queued,
            "active": self.active,
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "worker_count": self.worker_count,
            "closed": self.closed,
        }


@dataclass
class _Job(Generic[T]):
    future: Future[T]
    function: Callable[..., T]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


_STOP = object()


class InferenceCoordinator:
    """A bounded FIFO dispatcher with exactly one inference worker.

    QR detection and OCR libraries are not reliably re-entrant.  A single
    coordinator is shared by all camera stations in the UI service, so their
    expensive vision work is serialized without serializing HTTP, capture, or
    local-storage work.
    """

    def __init__(self, max_queue_size: int = 8, *, name: str = "roll-inference") -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        self.max_queue_size = int(max_queue_size)
        self._queue: queue.Queue[_Job[Any] | object] = queue.Queue(self.max_queue_size)
        self._state_lock = threading.Lock()
        self._closed = False
        self._active = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._worker = threading.Thread(target=self._work, name=name, daemon=True)
        self._worker.start()

    @property
    def worker_count(self) -> int:
        """The configured number of workers (intentionally always one)."""

        return 1

    def submit(
        self,
        function: Callable[..., T],
        /,
        *args: Any,
        block: bool = False,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Future[T]:
        future: Future[T] = Future()
        with self._state_lock:
            if self._closed:
                raise InferenceCoordinatorClosed("inference coordinator is closed")
            self._submitted += 1
        job: _Job[T] = _Job(future, function, args, kwargs)
        try:
            self._queue.put(job, block=block, timeout=timeout if block else None)
        except queue.Full as exc:
            with self._state_lock:
                self._submitted -= 1
            raise InferenceQueueFull("inference queue is full") from exc
        return future

    def run(
        self,
        function: Callable[..., T],
        /,
        *args: Any,
        queue_timeout: float | None = None,
        result_timeout: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Submit one job and wait for it, preserving FIFO order."""

        future = self.submit(
            function,
            *args,
            block=queue_timeout is not None,
            timeout=queue_timeout,
            **kwargs,
        )
        return future.result(timeout=result_timeout)

    def status(self) -> InferenceQueueStatus:
        with self._state_lock:
            return InferenceQueueStatus(
                capacity=self.max_queue_size,
                queued=self._queue.qsize(),
                active=self._active,
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                worker_count=1,
                closed=self._closed,
            )

    def close(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        with self._state_lock:
            if self._closed:
                if wait and self._worker.is_alive():
                    self._worker.join()
                return
            self._closed = True
        if cancel_pending:
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, _Job):
                    pending.future.cancel()
                self._queue.task_done()
        # close() is allowed to wait for existing work, so the stop sentinel may
        # wait for a queue slot rather than bypassing FIFO ordering.
        self._queue.put(_STOP)
        if wait:
            self._worker.join()

    shutdown = close

    def __enter__(self) -> "InferenceCoordinator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _Job)
                if not item.future.set_running_or_notify_cancel():
                    continue
                with self._state_lock:
                    self._active = True
                try:
                    result = item.function(*item.args, **item.kwargs)
                except BaseException as exc:
                    with self._state_lock:
                        self._failed += 1
                    item.future.set_exception(exc)
                else:
                    with self._state_lock:
                        self._completed += 1
                    item.future.set_result(result)
                finally:
                    with self._state_lock:
                        self._active = False
            finally:
                self._queue.task_done()


# Explicit descriptive alias used by callers/tests that prefer the longer name.
BoundedFIFOInferenceCoordinator = InferenceCoordinator
