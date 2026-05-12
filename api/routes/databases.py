from fastapi import APIRouter, Depends

from api.deps import get_milvus_client, get_milvus_service
from api.schemas import (
    CreateDatabaseRequest,
    DropDatabaseRequest,
    ListDatabasesResponse,
    MessageResponse,
    TruncateDatabaseRequest,
    UsingDatabaseRequest,
)
from milvus.client import MilvusClient
from retrieval.service import MilvusService

router = APIRouter()


@router.get("", response_model=ListDatabasesResponse)
def list_databases(client: MilvusClient = Depends(get_milvus_client)):
    dbs = client.list_databases()
    return ListDatabasesResponse(databases=dbs)


@router.post("", response_model=MessageResponse)
def create_database(
    req: CreateDatabaseRequest,
    client: MilvusClient = Depends(get_milvus_client),
):
    client.create_database(req.db_name)
    return MessageResponse(message=f"Database '{req.db_name}' created")


@router.delete("", response_model=MessageResponse)
def drop_database(
    req: DropDatabaseRequest,
    client: MilvusClient = Depends(get_milvus_client),
):
    client.drop_database(req.db_name)
    return MessageResponse(message=f"Database '{req.db_name}' dropped")


@router.put("/using", response_model=MessageResponse)
def using_database(
    req: UsingDatabaseRequest,
    client: MilvusClient = Depends(get_milvus_client),
):
    client.using_database(req.db_name)
    return MessageResponse(message=f"Switched to database '{req.db_name}'")


@router.post("/truncate", response_model=MessageResponse)
def truncate_database(
    req: TruncateDatabaseRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    original_db = service.client.database
    service.client.using_database(req.database)
    try:
        collections = service.client.list_collections()
        total_deleted = 0
        for col_name in collections:
            total_deleted += service.truncate_collection(collection_name=col_name)
    finally:
        service.client.using_database(original_db)
    return MessageResponse(
        message=f"Truncated database '{req.database}': {len(collections)} collections, {total_deleted} rows deleted"
    )
