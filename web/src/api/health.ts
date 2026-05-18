import axios from 'axios';
import type { HealthResponse } from './types';

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await axios.get<HealthResponse>('/health/');
  return data;
}
