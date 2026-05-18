import apiClient from './client';
import type {
  CreateCollectionRequest,
  DropCollectionRequest,
  DescribeCollectionResponse,
} from './types';

export async function listCollections(database?: string): Promise<string[]> {
  const { data } = await apiClient.get<string[]>('/collections', {
    params: database ? { database } : undefined,
  });
  return data.sort();
}

export async function createCollection(req: CreateCollectionRequest): Promise<void> {
  await apiClient.post('/collections/create', req);
}

export async function dropCollection(req: DropCollectionRequest): Promise<void> {
  await apiClient.post('/collections/drop', req);
}

export async function describeCollection(
  name: string,
  database?: string,
): Promise<DescribeCollectionResponse> {
  const { data } = await apiClient.get<DescribeCollectionResponse>(
    `/collections/${name}/schema`,
    { params: database ? { database } : undefined },
  );
  return data;
}
