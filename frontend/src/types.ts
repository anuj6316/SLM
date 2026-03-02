// My custom types
export interface HealthRequest {
  groq_api_key: string;
  jina_api_key: string;
}
export interface HealthResponse {
  groq_isActive: boolean;
  jina_isAtive: boolean;
}
// ---

export interface ServiceStatus {
  jina: 'online' | 'offline';
  groq: 'online' | 'offline';
}

export interface PipelineStep {
  id: string;
  status: 'complete' | 'active' | 'pending' | 'queued' | 'error';
}

export interface PipelineStatus {
  status: 'idle' | 'active' | 'error';
  currentStepId: string | null;
  steps: PipelineStep[];
}

export interface SystemStatus {
  services: ServiceStatus;
  pipeline: PipelineStatus;
}

export interface Metric {
  id: string;
  label: string;
  value: string;
  trend: number[];
  color?: string; // Optional, for UI mapping
  isBar?: boolean; // Optional, for UI mapping
}

export interface MetricsOverview {
  metrics: Metric[];
}

export interface LogEntry {
  id: string;
  timestamp: string;
  type: 'INFO' | 'ERROR' | 'WARNING';
  scope: string;
  message: string;
}

export interface LogsResponse {
  logs: LogEntry[];
}

export interface QualityDataPoint {
  name: string;
  value: number;
}

export interface QualityDistribution {
  period: string;
  data: QualityDataPoint[];
}

export interface Dataset {
  id: string | number;
  name: string;
  type: string;
  created: string;
  createdRelative: string;
}

export interface RecentDatasets {
  datasets: Dataset[];
}

export interface RunPipelineRequest {
  configId: string;
  forceRestart?: boolean;
}

export interface RunPipelineResponse {
  success: boolean;
  pipelineId: string;
  message: string;
}

export interface HealthRequest {
  groq_api_key: string;
  jina_api_key: string;
}

export interface HealthResponse {
  groq_isActive: boolean;
  jina_isActive: boolean;
}
