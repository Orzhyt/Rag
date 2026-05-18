import apiClient from './client';
import type { ListDatabasesResponse, CreateDatabaseRequest, DropDatabaseRequest } from './types';

export async function listDatabases(): Promise<string[]> {
  const { data } = await apiClient.get<ListDatabasesResponse>('/databases');
  return data.databases.sort();
}

export async function createDatabase(req: CreateDatabaseRequest): Promise<void> {
  await apiClient.post('/databases', req);
}

export async function dropDatabase(req: DropDatabaseRequest): Promise<void> {
  await apiClient.delete('/databases', { data: req });
}
