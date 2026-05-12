from pathlib import Path

from fastapi import APIRouter, Depends

from api.deps import get_milvus_service
from api.schemas import (
    DeleteByChunkIdsRequest,
    DeleteByFieldRequest,
    DeleteBySourceRequest,
    DeleteResponse,
    IngestRequest,
    IngestResponse,
)
from common.logger import setup_logger
from data_pipeline import parse_directory
from retrieval.service import MilvusService

logger = setup_logger("api.data")
router = APIRouter()


def _switch_db(service: MilvusService, database):
    if database and database != service.client.database:
        service.client.using_database(database)


@router.post("/ingestion", response_model=IngestResponse)
def ingest_directory(
    req: IngestRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    folder = Path(req.folder_path)
    if not folder.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {req.folder_path}")

    _switch_db(service, req.database)

    if req.collection_name is not None:
        service.client.collection_name = req.collection_name

    col_name = service.client.collection_name
    if not service.client.has_collection(col_name):
        raise ValueError(f"Collection '{col_name}' does not exist. Create it first via /collections/create.")

    chunks = parse_directory(
        req.folder_path,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )

    files_scanned = len({c.source_file for c in chunks})
    chunks_parsed = len(chunks)

    if not chunks:
        return IngestResponse(files_scanned=files_scanned, chunks_parsed=0, chunks_inserted=0)

    if req.upsert_mode:
        inserted = service.upsert(chunks)
    else:
        inserted = service.insert(chunks)

    logger.info(
        "Ingestion complete: %d files, %d chunks parsed, %d chunks inserted",
        files_scanned, chunks_parsed, inserted,
    )

    return IngestResponse(files_scanned=files_scanned, chunks_parsed=chunks_parsed, chunks_inserted=inserted)


@router.post("/delete/chunk-ids", response_model=DeleteResponse)
def delete_by_chunk_ids(
    req: DeleteByChunkIdsRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service, req.database)
    n = service.delete_by_chunk_ids(req.chunk_ids, collection_name=req.collection_name)
    return DeleteResponse(deleted_count=n)


@router.post("/delete/source", response_model=DeleteResponse)
def delete_by_source(
    req: DeleteBySourceRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service, req.database)
    n = service.delete_by_source(req.source_file, collection_name=req.collection_name)
    return DeleteResponse(deleted_count=n)


@router.post("/delete/field", response_model=DeleteResponse)
def delete_by_field(
    req: DeleteByFieldRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service, req.database)
    n = service.delete_by_field(
        field_name=req.field_name,
        field_value=req.field_value,
        collection_name=req.collection_name,
    )
    return DeleteResponse(deleted_count=n)
