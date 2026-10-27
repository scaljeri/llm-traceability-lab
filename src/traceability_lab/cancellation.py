"""Cancellation belongs to one background execution, never another model request."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

_current = ContextVar("execution_cancellation", default=None)


class RunCancelled(Exception):
    pass


@contextmanager
def cancellation_scope(event):
    token = _current.set(event)
    try:
        yield
    finally:
        _current.reset(token)


def cancellation_event():
    return _current.get()


def check_cancelled():
    event = cancellation_event()
    if event is not None and event.is_set():
        raise RunCancelled("Cancelled by user; active model request aborted.")


async def cancellable_post(settings, body):
    """Cancelling the awaited request closes its socket via AsyncClient cleanup."""
    check_cancelled()
    async with httpx.AsyncClient(
        timeout=settings.timeout_seconds, trust_env=False, follow_redirects=False
    ) as client:
        request = asyncio.create_task(client.post(settings.llm_url + "/chat/completions", json=body))
        try:
            while not request.done():
                check_cancelled()
                await asyncio.wait({request}, timeout=0.05)
            check_cancelled()
            response = await request
            response.raise_for_status()
            return response.json()
        finally:
            if not request.done():
                request.cancel()
            # Always retrieve exceptions, including a response racing with Cancel.
            await asyncio.gather(request, return_exceptions=True)


async def cancellable_get(url, params):
    """Runtime metadata is also interruptible; it never performs generation."""
    check_cancelled()
    async with httpx.AsyncClient(timeout=3, trust_env=False, follow_redirects=False) as client:
        request = asyncio.create_task(client.get(url, params=params))
        try:
            while not request.done():
                check_cancelled()
                await asyncio.wait({request}, timeout=0.05)
            check_cancelled()
            response = await request
            response.raise_for_status()
            return response
        finally:
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)
