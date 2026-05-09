from fastapi import APIRouter, Depends

from api.deps import get_milvus_client
from api.schemas import (
    CreateDatabaseRequest,
    DropDatabaseRequest,
    ListDatabasesResponse,
    MessageResponse,
    UsingDatabaseRequest,
)
from milvus.client import MilvusClient

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
