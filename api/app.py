from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymilvus.exceptions import MilvusException

from api.deps import cleanup_singletons, init_singletons
from api.schemas import ErrorResponse
from api.routes import collections, databases, health, ingestion, search
from common.logger import setup_logger

logger = setup_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing singletons...")
    init_singletons()
    logger.info("Application started")
    yield
    logger.info("Shutting down...")
    cleanup_singletons()
    logger.info("Application stopped")


app = FastAPI(
    title="RAG Project API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Exception Handlers ───


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error=str(exc)).model_dump(),
    )


@app.exception_handler(MilvusException)
async def milvus_error_handler(request: Request, exc: MilvusException):
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(error="Milvus error", detail=str(exc)).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error").model_dump(),
    )


# ─── Routers ───

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])
app.include_router(databases.router, prefix="/api/v1/databases", tags=["Databases"])
app.include_router(collections.router, prefix="/api/v1/collections", tags=["Collections"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
