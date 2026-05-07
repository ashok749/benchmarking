from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from .config import Settings
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ErrorBody,
    ErrorResponse,
    ModelCard,
    ModelListResponse,
    Usage,
)
from .strategies import StrategyRunner


settings = Settings.load()
runner = StrategyRunner(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="DSPy OpenAI Endpoint", version="0.1.0", lifespan=lifespan)


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.endpoint_api_key:
        return
    expected = f"Bearer {settings.endpoint_api_key}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.exception_handler(FileNotFoundError)
async def handle_missing_artifact(_, exc: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(error=ErrorBody(message=str(exc), code="artifact_missing")).model_dump(),
    )


@app.exception_handler(ValueError)
async def handle_value_error(_, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error=ErrorBody(message=str(exc), code="bad_request")).model_dump(),
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "models": sorted(settings.model_aliases.keys())}


@app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(require_auth)])
async def list_models():
    return ModelListResponse(data=[ModelCard(id=alias) for alias in sorted(settings.model_aliases.keys())])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(require_auth)])
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        raise ValueError("Streaming is not implemented in this scaffold. Send stream=false.")

    model_alias = settings.model_aliases.get(request.model)
    if model_alias is None:
        raise ValueError(
            f"Unknown model alias '{request.model}'. Available models: {', '.join(sorted(settings.model_aliases))}"
        )

    result = await run_in_threadpool(runner.run, model_alias, request)
    prompt_tokens = sum(len((message.content if isinstance(message.content, str) else "").split()) for message in request.messages)
    completion_text = result.content or ""
    completion_tokens = len(completion_text.split())

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=result.content, tool_calls=result.tool_calls),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
