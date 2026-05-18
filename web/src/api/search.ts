import apiClient from './client';
import type { SearchRequest, SearchResponse } from './types';

export async function hybridSearch(req: SearchRequest): Promise<SearchResponse> {
  const { data } = await apiClient.post<SearchResponse>('/search/hybrid', req);
  return data;
}
