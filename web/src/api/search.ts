import apiClient from './client';
import type { SearchRequest, HybridSearchRequest, SearchResponse } from './types';

export async function vectorSearch(req: SearchRequest): Promise<SearchResponse> {
  const { data } = await apiClient.post<SearchResponse>('/search/vector', req);
  return data;
}

export async function hybridSearch(req: HybridSearchRequest): Promise<SearchResponse> {
  const { data } = await apiClient.post<SearchResponse>('/search/hybrid', req);
  return data;
}
