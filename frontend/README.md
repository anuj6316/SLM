# QA Data Pipeline Dashboard

A dashboard for managing unstructured data to QA pipelines with scraping, refinement, and AI generation monitoring.

## API Specification

This document outlines the API endpoints required to power the frontend dashboard with real data.

### 1. System & Pipeline Status
Controls the header status indicators and the main pipeline progress flow.

**GET /api/status**
Returns the health of external services and the current state of the pipeline steps.

**Response:**
```json
{
  "services": {
    "jina": "online",
    "groq": "online"
  },
  "pipeline": {
    "status": "active", // idle, active, error
    "currentStepId": "scraping",
    "steps": [
      { "id": "input", "status": "complete" },
      { "id": "scraping", "status": "active" },
      { "id": "refinement", "status": "pending" },
      { "id": "aigen", "status": "queued" },
      { "id": "output", "status": "queued" }
    ]
  }
}
```

### 2. Run Pipeline
Triggers the execution of the data pipeline.

**POST /api/pipeline/run**

**Request:**
```json
{
  "configId": "default-v1",
  "forceRestart": true
}
```

**Response:**
```json
{
  "success": true,
  "pipelineId": "run_123456789",
  "message": "Pipeline started successfully"
}
```

### 3. Dashboard Metrics
Populates the top row of sparkline cards.

**GET /api/metrics/overview**

**Response:**
```json
{
  "metrics": [
    {
      "id": "total_urls",
      "label": "Total URLs",
      "value": "206",
      "trend": [10, 15, 12, 20, 25, 22, 30]
    },
    {
      "id": "chunks",
      "label": "Chunks",
      "value": "35,373",
      "trend": [50, 60, 55, 70, 65, 80, 75]
    },
    {
      "id": "qa_pairs",
      "label": "QA Pairs",
      "value": "208",
      "trend": [5, 8, 12, 10, 15, 20, 25]
    },
    {
      "id": "avg_quality",
      "label": "Avg Quality",
      "value": "40.9%",
      "trend": [30, 35, 32, 38, 40, 42, 41]
    },
    {
      "id": "token_usage",
      "label": "Token Usage",
      "value": "22 MB",
      "trend": [10, 20, 15, 25, 30, 28, 35]
    }
  ]
}
```

### 4. Process Logs
Feeds the "Active Process Log" terminal view.

**GET /api/logs**
*Query Params: `?limit=50&since=timestamp`*

**Response:**
```json
{
  "logs": [
    {
      "id": "log_1",
      "timestamp": "2024-03-20T10:00:01Z",
      "type": "INFO",
      "scope": "SCRAPE",
      "message": "Scraping QAxFoam.org"
    },
    {
      "id": "log_2",
      "timestamp": "2024-03-20T10:00:02Z",
      "type": "ERROR",
      "scope": "JUDGE",
      "message": "CompleterInterlath saric Error..."
    }
  ]
}
```

### 5. Quality Distribution
Populates the bar chart showing quality metrics over time.

**GET /api/metrics/quality**

**Response:**
```json
{
  "period": "weekly",
  "data": [
    { "name": "Mon", "value": 40 },
    { "name": "Tue", "value": 100 },
    { "name": "Wed", "value": 60 },
    { "name": "Thu", "value": 80 },
    { "name": "Fri", "value": 20 },
    { "name": "Sat", "value": 20 }
  ]
}
```

### 6. Recent Datasets
Populates the list of generated datasets in the bottom right.

**GET /api/datasets/recent**

**Response:**
```json
{
  "datasets": [
    {
      "id": "ds_1",
      "name": "Unstructured to QA (Batch A)",
      "type": "JetBrains Mono", // or "JSON", "CSV" etc.
      "created": "2024-03-20T08:00:00Z",
      "createdRelative": "2 hours ago"
    },
    {
      "id": "ds_2",
      "name": "Unstructured to QA (Batch B)",
      "type": "JetBrains Mono",
      "created": "2024-03-20T07:00:00Z",
      "createdRelative": "3 hours ago"
    }
  ]
}
```
