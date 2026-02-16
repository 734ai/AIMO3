# AIMO3 Solver - Production API Documentation

## Overview

The AIMO3 Solver provides a production-ready REST API for solving mathematical olympiad problems. This document describes all endpoints, authentication, error handling, and best practices.

---

## Base URL

```
Production:  https://api.aimo3.example.com/v1
Staging:     https://staging-api.aimo3.example.com/v1
Development: http://localhost:8000/v1
```

---

## Authentication

All API requests require authentication using an API key in the `Authorization` header.

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.aimo3.example.com/v1/health
```

**Getting an API Key:**
1. Create an account at https://aimo3.example.com
2. Navigate to Settings → API Keys
3. Generate a new API key
4. Store it securely in your `.env` file

**API Key Rotation:**
- Rotate keys every 90 days
- Generate new key before rotating
- Test with new key before deactivating old one

---

## Rate Limiting

**Default Limits:**
- 100 requests per minute
- 1000 requests per hour

**Handling Rate Limits:**
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 12
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1695385500
```

---

## Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check API health status

**Request:**
```bash
curl https://api.aimo3.example.com/v1/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "model": "healthy",
    "cache": "healthy",
    "database": "healthy"
  }
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "error": "Model failed to load",
  "components": {
    "model": "unhealthy",
    "cache": "healthy"
  }
}
```

---

### 2. Solve Single Problem

**Endpoint:** `POST /solve`

**Description:** Solve a single mathematical problem

**Request:**
```bash
curl -X POST https://api.aimo3.example.com/v1/solve \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Find the value of $2^3 + 3^2$",
    "problem_id": "P001",
    "timeout_seconds": 30
  }'
```

**Request Body:**
```json
{
  "problem": "string (required)",           // Problem text or LaTeX
  "problem_id": "string (optional)",        // Unique problem identifier
  "timeout_seconds": "integer (optional)",  // Max execution time (default: 30)
  "metadata": {                             // Optional metadata
    "source": "kaggle",
    "difficulty": "medium"
  }
}
```

**Response (200 OK):**
```json
{
  "problem_id": "P001",
  "answer": 17,
  "confidence": 0.95,
  "reasoning": "Step 1: Calculate 2^3 = 8\nStep 2: Calculate 3^2 = 9\nStep 3: Add them: 8 + 9 = 17",
  "duration_ms": 234,
  "model_used": "Open-Orca/orca_mini_3b",
  "verification": {
    "status": "verified",
    "method": "symbolic"
  },
  "timestamp": "2024-01-15T10:30:45Z"
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Invalid input",
  "details": "Problem must be between 10 and 10000 characters",
  "error_code": "INVALID_PROBLEM"
}
```

**Response (408 Request Timeout):**
```json
{
  "error": "Problem solving timeout",
  "problem_id": "P001",
  "duration_ms": 30000,
  "error_code": "TIMEOUT",
  "partial_result": {
    "reasoning": "Step 1: ...",
    "status": "incomplete"
  }
}
```

---

### 3. Batch Solve Problems

**Endpoint:** `POST /batch/solve`

**Description:** Solve multiple problems in batch

**Request:**
```bash
curl -X POST https://api.aimo3.example.com/v1/batch/solve \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "problems": [
      {
        "problem_id": "P001",
        "problem": "What is 2+2?"
      },
      {
        "problem_id": "P002",
        "problem": "What is 5*6?"
      }
    ],
    "parallel": true
  }'
```

**Request Body:**
```json
{
  "problems": [
    {
      "problem_id": "string (required)",
      "problem": "string (required)"
    }
  ],
  "parallel": "boolean (optional, default: true)",
  "batch_size": "integer (optional, default: 32)"
}
```

**Response (200 OK):**
```json
{
  "batch_id": "batch_1695385500",
  "status": "completed",
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "problem_id": "P001",
      "answer": 4,
      "confidence": 0.98,
      "status": "success"
    },
    {
      "problem_id": "P002",
      "answer": 30,
      "confidence": 0.99,
      "status": "success"
    }
  ],
  "duration_ms": 856
}
```

**Response (202 Accepted) - Async Processing:**
```json
{
  "batch_id": "batch_1695385500",
  "status": "processing",
  "status_url": "/v1/batch/status/batch_1695385500"
}
```

---

### 4. Check Batch Status

**Endpoint:** `GET /batch/status/{batch_id}`

**Description:** Check status of batch job

**Request:**
```bash
curl https://api.aimo3.example.com/v1/batch/status/batch_1695385500 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "batch_id": "batch_1695385500",
  "status": "completed",
  "progress": {
    "completed": 2,
    "total": 2,
    "percentage": 100
  },
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:31:00Z",
  "results_url": "/v1/batch/results/batch_1695385500"
}
```

---

### 5. Get Batch Results

**Endpoint:** `GET /batch/results/{batch_id}`

**Description:** Retrieve results from completed batch

**Request:**
```bash
curl https://api.aimo3.example.com/v1/batch/results/batch_1695385500 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "batch_id": "batch_1695385500",
  "results": [
    {
      "problem_id": "P001",
      "answer": 4,
      "confidence": 0.98,
      "status": "success"
    }
  ],
  "export_formats": {
    "csv": "/v1/batch/results/batch_1695385500/export/csv",
    "json": "/v1/batch/results/batch_1695385500/export/json"
  }
}
```

---

### 6. Statistics & Metrics

**Endpoint:** `GET /stats`

**Description:** Get API usage statistics

**Request:**
```bash
curl https://api.aimo3.example.com/v1/stats \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "period": "today",
  "requests_total": 1245,
  "requests_successful": 1198,
  "requests_failed": 47,
  "success_rate": 0.9623,
  "average_latency_ms": 234,
  "p50_latency_ms": 210,
  "p95_latency_ms": 450,
  "p99_latency_ms": 800,
  "cache_hit_rate": 0.42,
  "api_key_quota": {
    "limit": 10000,
    "used": 1245,
    "remaining": 8755
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "status_code": 400,
  "details": "Additional details",
  "request_id": "req_123456789",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_PROBLEM` | 400 | Problem format invalid |
| `INVALID_API_KEY` | 401 | API key missing or invalid |
| `UNAUTHORIZED` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `TIMEOUT` | 408 | Request timeout |
| `SERVER_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Best Practices

### 1. Request Handling

```python
import requests
import time

def solve_with_retry(problem, max_retries=3):
    """Solve problem with exponential backoff retry."""
    api_key = os.environ['AIMO3_API_KEY']
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.aimo3.example.com/v1/solve",
                json={"problem": problem},
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited, wait
                retry_after = int(response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                continue
            elif response.status_code >= 500:
                # Server error, retry
                time.sleep(2 ** attempt)
                continue
            else:
                # Client error, don't retry
                response.raise_for_status()
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception(f"Failed after {max_retries} attempts")
```

### 2. Batch Processing

```python
def batch_solve(problems, batch_size=32):
    """Solve problems in batches."""
    api_key = os.environ['AIMO3_API_KEY']
    headers = {"Authorization": f"Bearer {api_key}"}
    
    results = []
    
    for i in range(0, len(problems), batch_size):
        batch = problems[i:i+batch_size]
        
        response = requests.post(
            "https://api.aimo3.example.com/v1/batch/solve",
            json={"problems": batch},
            headers=headers
        )
        
        if response.status_code == 202:
            # Async processing
            batch_id = response.json()['batch_id']
            results.extend(poll_batch_status(batch_id))
        else:
            results.extend(response.json()['results'])
    
    return results
```

### 3. Caching

```python
def solve_with_caching(problem, cache_ttl=3600):
    """Solve with caching to reduce API calls."""
    cache_key = hashlib.md5(problem.encode()).hexdigest()
    
    # Check cache
    if cache.get(cache_key):
        return cache.get(cache_key)
    
    # Call API
    result = solve_with_retry(problem)
    
    # Cache result
    cache.set(cache_key, result, ttl=cache_ttl)
    
    return result
```

---

## Webhooks (Optional)

Configure webhooks for batch completion notifications:

```bash
curl -X POST https://api.aimo3.example.com/v1/webhooks \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://yoursite.example.com/webhooks/aimo3",
    "events": ["batch.completed"],
    "secret": "your_webhook_secret"
  }'
```

**Webhook Payload:**
```json
{
  "event": "batch.completed",
  "batch_id": "batch_1695385500",
  "timestamp": "2024-01-15T10:31:00Z",
  "results_url": "/v1/batch/results/batch_1695385500"
}
```

---

## Support & Issues

- **Documentation:** https://docs.aimo3.example.com
- **Status Page:** https://status.aimo3.example.com
- **Email Support:** support@aimo3.example.com
- **Slack Community:** [Join our Slack](https://slack.aimo3.example.com)

