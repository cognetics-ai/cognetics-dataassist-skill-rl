export type StatusType = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'idle';

export interface AuthMeResponse {
  soeid: string;
  role: string;
  display_name: string;
  email: string;
  job_title: string;
}

export interface DiscoveryQuery {
  query_id: string;
  sql_text: string;
  sql2text: string;
  tables: string[];
  created_at: string;
  engine: string;
}

export interface DiscoveryRoleContextResponse {
  soeid: string;
  role: string;
  queries: DiscoveryQuery[];
  metadata: Record<string, unknown>;
}

export interface DraftResponse {
  draft_sql: string;
  explanation: string;
  warnings: string[];
  context_refs: Array<Record<string, unknown>>;
  confidence: number;
  assumptions: string[];
}

export interface ValidateResponse {
  is_valid: boolean;
  policy_findings: Array<Record<string, unknown>>;
  explain_summary: Record<string, unknown>;
  risk_score: number;
  fixes: string[];
}

export interface RunResponse {
  run_id: string;
}

export interface ResultsResponse {
  run_id: string;
  status: StatusType;
  schema: Array<{ name: string; type: string }>;
  rows: Array<Array<unknown>>;
  next_page_token?: string | null;
  error_message?: string | null;
}

export interface QueryRunHistoryItem {
  run_id: string;
  soeid: string;
  engine: string;
  input_mode: string;
  route_mode?: string | null;
  submitted_text: string;
  submitted_sql?: string | null;
  submitted_prompt?: string | null;
  natural_language_query?: string | null;
  final_sql: string;
  status: StatusType;
  query_start_time?: string | null;
  query_end_time?: string | null;
  created_at: string;
  error_message?: string | null;
  row_count: number;
}

export interface QueryRunHistoryResponse {
  soeid: string;
  runs: QueryRunHistoryItem[];
}

export interface StreamEvent {
  event_id: string;
  run_id: string;
  event_type: string;
  timestamp: string;
  payload: Record<string, unknown>;
}
