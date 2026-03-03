import { 
  SystemStatus, 
  MetricsOverview, 
  LogsResponse, 
  QualityDistribution, 
  RecentDatasets,
  RunPipelineRequest,
  RunPipelineResponse,
  HealthRequest,
  HealthResponse
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Mock data to simulate API responses (keeping them for endpoints not yet implemented on backend)
const MOCK_DELAY = 500;

const mockSystemStatus: SystemStatus = {
  services: {
    jina: 'offline',
    groq: 'offline'
  },
  pipeline: {
    status: 'active',
    currentStepId: 'scraping',
    steps: [
      { id: 'input', status: 'complete' },
      { id: 'scraping', status: 'active' },
      { id: 'refinement', status: 'pending' },
      { id: 'aigen', status: 'queued' },
      { id: 'output', status: 'queued' }
    ]
  }
};

const mockMetrics: MetricsOverview = {
  metrics: [
    { id: 'total_urls', label: 'Total URLs', value: '206', trend: [10, 15, 12, 20, 25, 22, 30] },
    { id: 'chunks', label: 'Chunks', value: '35,373', trend: [50, 60, 55, 70, 65, 80, 75] },
    { id: 'qa_pairs', label: 'QA Pairs', value: '208', trend: [5, 8, 12, 10, 15, 20, 25] },
    { id: 'avg_quality', label: 'Avg Quality', value: '40.9%', trend: [30, 35, 32, 38, 40, 42, 41] },
    { id: 'token_usage', label: 'Token Usage', value: '22 MB', trend: [10, 20, 15, 25, 30, 28, 35], isBar: true }
  ]
};

const mockLogs: LogsResponse = {
  logs: [
    { id: '1', timestamp: new Date().toISOString(), type: 'INFO', scope: 'SCRAPE', message: 'Scraping QAxFoam.org' },
    { id: '2', timestamp: new Date().toISOString(), type: 'INFO', scope: 'SCRAPE', message: 'Collecting analfsrsurreer.inam.com...' },
    { id: '12', timestamp: new Date().toISOString(), type: 'ERROR', scope: 'JUDGE', message: 'CompleterInterlath saric Error...' },
  ]
};

const mockQuality: QualityDistribution = {
  period: 'weekly',
  data: [
    { name: 'Mon', value: 40 },
    { name: 'Tue', value: 100 },
    { name: 'Wed', value: 60 },
    { name: 'Thu', value: 80 },
    { name: 'Fri', value: 20 },
    { name: 'Sat', value: 20 },
  ]
};

const mockDatasets: RecentDatasets = {
  datasets: [
    { id: 1, name: 'Unstructured to QA (Batch A)', type: 'JetBrains Mono', created: new Date().toISOString(), createdRelative: '2 hours ago' },
  ]
};

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const api = {
  checkHealth: async (request: HealthRequest): Promise<HealthResponse> => {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API health check failed: ${response.statusText}`);
    }

    return response.json();
  },

  getStatus: async (): Promise<SystemStatus> => {
    await sleep(MOCK_DELAY);
    return mockSystemStatus;
  },

  getMetrics: async (): Promise<MetricsOverview> => {
    await sleep(MOCK_DELAY);
    return mockMetrics;
  },

  getLogs: async (): Promise<LogsResponse> => {
    await sleep(MOCK_DELAY);
    return mockLogs;
  },

  getQualityDistribution: async (): Promise<QualityDistribution> => {
    await sleep(MOCK_DELAY);
    return mockQuality;
  },

  getRecentDatasets: async (): Promise<RecentDatasets> => {
    await sleep(MOCK_DELAY);
    return mockDatasets;
  },

  login: async (credentials: any) => {
    const response = await fetch(`${API_BASE_URL}/account/login/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Login failed');
    }

    return response.json();
  },

  signup: async (userData: any) => {
    const response = await fetch(`${API_BASE_URL}/account/signup/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(JSON.stringify(errorData) || 'Signup failed');
    }

    return response.json();
  },

  runPipeline: async (request: RunPipelineRequest): Promise<RunPipelineResponse> => {
    await sleep(MOCK_DELAY * 2);
    return {
      success: true,
      pipelineId: `run_${Date.now()}`,
      message: 'Pipeline started successfully'
    };
  }
};
