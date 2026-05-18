import apiClient from './client';
import type { IngestRequest, IngestResponse } from './types';

export async function ingestData(req: IngestRequest): Promise<IngestResponse> {
  const { data } = await apiClient.post<IngestResponse>('/data/ingestion', req);
  return data;
}
