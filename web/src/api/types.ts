// ─── Health ───

export interface HealthResponse {
  status: string;
  milvus_connected: boolean;
  embedding_model_loaded: boolean;
  embedding_device: string;
}

// ─── Databases ───

export interface ListDatabasesResponse {
  databases: string[];
}

export interface CreateDatabaseRequest {
  db_name: string;
}

export interface DropDatabaseRequest {
  db_name: string;
}

// ─── Collections ───

export interface FieldSpec {
  name: string;
  dtype: string;
  is_primary?: boolean;
  auto_id?: boolean;
  max_length?: number;
  dim?: number;
  description?: string;
}

export interface CreateCollectionRequest {
  dim?: number;
  drop_if_exists?: boolean;
  collection_name?: string;
  fields?: FieldSpec[];
  description?: string;
  database?: string;
}

export interface DropCollectionRequest {
  collection_name?: string;
  database?: string;
}

export interface DescribeCollectionResponse {
  name: string;
  description: string;
  fields: Record<string, unknown>[];
}

// ─── Data Ingestion ───

export interface IngestRequest {
  folder_path: string;
  upsert_mode?: boolean;
  collection_name?: string;
  database?: string;
}

export interface IngestResponse {
  files_scanned: number;
  chunks_parsed: number;
  chunks_inserted: number;
}

// ─── Search ───

export interface SearchRequest {
  query: string;
  filter_expr?: string;
  output_fields?: string[];
  collection_names?: string[];
  database?: string;
}

export interface SearchHit {
  score: number;
  [key: string]: unknown;
}

export interface SearchResponse {
  results: SearchHit[];
  total: number;
}

// ─── Chat ───

export interface SourceCitation {
  index: number;
  chunk_id: string;
  source_file: string;
  score: number;
  content: string;
  fields?: Record<string, unknown>;
}

export interface ChatRequest {
  query: string;
  conversation_id?: string;
  collection_names?: string[];
  database?: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceCitation[];
  conversation_id: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources?: SourceCitation[];
}

export interface Conversation {
  id: string;
  title?: string;
  created_at: number;
  message_count: number;
  messages?: ChatMessage[];
}
