from pathlib import Path

from fastapi import APIRouter, Depends

from api.deps import get_milvus_service
from api.schemas import IngestRequest, IngestResponse
from common.logger import setup_logger
from data_pipeline import parse_directory
from retrieval.service import MilvusService

logger = setup_logger("api.ingestion")
router = APIRouter()


@router.post("", response_model=IngestResponse)
def ingest_directory(
    req: IngestRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    folder = Path(req.folder_path)
    if not folder.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {req.folder_path}")

    # 切换到指定数据库
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)

    try:
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
    finally:
        # 恢复原始数据库
        if original_db is not None:
            service.client.using_database(original_db)

    return IngestResponse(files_scanned=files_scanned, chunks_parsed=chunks_parsed, chunks_inserted=inserted)
