# PRODUCTION DEPLOYMENT GUIDE

## Overview

This guide provides comprehensive instructions for deploying the AIMO3 Solver to production environments with enterprise-grade reliability, monitoring, and security.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Architectures](#deployment-architectures)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security Hardening](#security-hardening)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Rollback Procedures](#rollback-procedures)

---

## Prerequisites

### System Requirements

- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB minimum (16GB+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, for faster inference)
- **Storage**: 20GB for models, 10GB for logs/cache
- **Python**: 3.8 to 3.11
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2

### Required Tools

```bash
# Python environment
python --version  # Verify Python 3.8+

# Package managers
pip install --upgrade pip setuptools wheel

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows
```

---

## Pre-Deployment Checklist

### 1. Code Quality Checks

```bash
# Run tests
pytest tests/ -v --cov=src

# Type checking
mypy src/ --strict

# Linting
flake8 src/ --max-line-length=100
pylint src/

# Code formatting
black src/ --check
isort src/ --check-only
```

### 2. Security Validation

```bash
# Check for security issues
bandit -r src/

# Validate credentials are not in code
grep -r "KAGGLE_API_KEY\|API_KEY\|SECRET" src/ config/

# Verify .env is in .gitignore
grep ".env" .gitignore
```

### 3. Performance Baseline

```bash
# Run performance tests
pytest tests/ -k performance -v

# Profile memory usage
python -m memory_profiler src/pipeline.py

# Benchmark key functions
python src/benchmark.py
```

### 4. Configuration Validation

```bash
# Verify all required environment variables are documented
grep "os.environ\|getenv" src/*.py

# Check config file integrity
python -c "from src.config import *; print('Config OK')"

# Validate against schema
python scripts/validate_config.py
```

---

## Deployment Architectures

### Architecture 1: Single Server (Development/Testing)

```
┌─────────────────────────────────────────┐
│           Single Server                  │
├─────────────────────────────────────────┤
│  - Application (src/)                   │
│  - Cache (in-memory)                    │
│  - Logs (local files)                   │
│  - Database (local SQLite)              │
└─────────────────────────────────────────┘
```

**Use When**: Learning, testing, low traffic (< 10 requests/min)

**Deployment**:
```bash
cd /home/hssn/Documents/kaggle/ai|mo
python -m src.pipeline  # or via Kaggle notebook
```

### Architecture 2: Multi-Server with Load Balancing (Production)

```
┌─────────────────────────────────────────┐
│         Load Balancer (Nginx)           │
└────────┬────────────────────────┬───────┘
         │                        │
    ┌────▼─────┐         ┌────────▼───┐
    │ Server 1  │         │  Server 2   │
    │ App + WS  │         │  App + WS   │
    └────┬─────┘         └────────┬───┘
         │                        │
    ┌────▼────────────────────────▼───┐
    │   Distributed Cache (Redis)     │
    ├────────────────────────────────┤
    │   External Monitoring (Prometheus)
    │   Log Aggregation (ELK Stack)
    │   Alert Manager (PagerDuty)
    └────────────────────────────────┘
```

**Use When**: Production with concurrent users, high availability required

**Components**:
- Load Balancer: Nginx or HAProxy
- Cache: Redis for distributed caching
- Monitoring: Prometheus + Grafana
- Logging: ELK Stack or Datadog

---

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/
COPY setup.py .

# Install package
RUN pip install -e .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.monitoring import HealthCheck; HealthCheck().run_all()"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "src.pipeline"]
```

### 2. Build and Run Docker Image

```bash
# Build
docker build -t aimo3-solver:latest .

# Run with GPU support
docker run --gpus all \
    -e KAGGLE_API_KEY="$KAGGLE_API_KEY" \
    -e KAGGLE_USERNAME="$KAGGLE_USERNAME" \
    -v /data:/app/data \
    -v /logs:/app/logs \
    -p 8000:8000 \
    aimo3-solver:latest

# Run with resource limits
docker run --gpus all \
    --memory=8g \
    --cpus=4 \
    -e KAGGLE_API_KEY="$KAGGLE_API_KEY" \
    aimo3-solver:latest
```

### 3. Docker Compose for Full Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    container_name: aimo3-app
    environment:
      - KAGGLE_API_KEY=${KAGGLE_API_KEY}
      - KAGGLE_USERNAME=${KAGGLE_USERNAME}
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      - redis
    networks:
      - aimo-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import src.pipeline"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: aimo3-cache
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - aimo-network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: aimo3-prometheus
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - aimo-network
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:

networks:
  aimo-network:
    driver: bridge
```

---

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aimo3-solver
  namespace: aimo3
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aimo3-solver
  template:
    metadata:
      labels:
        app: aimo3-solver
    spec:
      containers:
      - name: aimo3-app
        image: aimo3-solver:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "4"
            memory: "8Gi"
            nvidia.com/gpu: "1"
        env:
        - name: KAGGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: kaggle-credentials
              key: api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: aimo3-service
  namespace: aimo3
spec:
  type: LoadBalancer
  selector:
    app: aimo3-solver
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: aimo3-config
  namespace: aimo3
data:
  config.yaml: |
    model:
      name: "Open-Orca/orca_mini_3b"
      device: "auto"
    cache:
      ttl_seconds: 3600
      max_size: 10000
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace aimo3

# Create secrets
kubectl create secret generic kaggle-credentials \
  --from-literal=api-key=$KAGGLE_API_KEY \
  -n aimo3

# Apply manifests
kubectl apply -f k8s/deployment.yaml

# Verify deployment
kubectl get pods -n aimo3
kubectl logs -n aimo3 deployment/aimo3-solver

# Port forward for testing
kubectl port-forward -n aimo3 svc/aimo3-service 8000:8000
```

---

## Monitoring & Observability

### 1. Health Checks

```python
# src/health.py
from src.monitoring import HealthCheck, HealthStatus, HealthCheckResult

def health_endpoint():
    """FastAPI health endpoint."""
    hc = HealthCheck()
    
    # Register checks
    hc.register("model", check_model_loaded)
    hc.register("cache", check_cache_available)
    hc.register("resources", check_system_resources)
    
    results = hc.run_all()
    
    # Determine overall status
    statuses = [r.status for r in results.values()]
    if any(s == HealthStatus.UNHEALTHY for s in statuses):
        return {"status": "unhealthy"}, 503
    elif any(s == HealthStatus.DEGRADED for s in statuses):
        return {"status": "degraded"}, 200
    else:
        return {"status": "healthy"}, 200

def check_model_loaded():
    """Check if model is loaded."""
    try:
        # Check model is available
        return HealthCheckResult(
            component="model",
            status=HealthStatus.HEALTHY,
            message="Model loaded successfully"
        )
    except Exception as e:
        return HealthCheckResult(
            component="model",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )

def check_system_resources():
    """Check system resources."""
    from src.monitoring import ResourceMonitor
    
    report = ResourceMonitor.get_full_report()
    
    # Check thresholds
    cpu_percent = report["cpu"]["system_percent"]
    memory_percent = report["memory"]["system_memory_percent"]
    
    if cpu_percent > 90 or memory_percent > 85:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.HEALTHY
    
    return HealthCheckResult(
        component="resources",
        status=status,
        message=f"CPU: {cpu_percent}%, Memory: {memory_percent}%",
        metadata=report
    )
```

### 2. Metrics Collection

```python
# src/metrics.py
from src.monitoring import MetricsCollector, PerformanceProfiler

# Global collector
metrics = MetricsCollector()

def record_inference_metrics(duration_ms, success=True):
    """Record inference metrics."""
    metrics.record("inference_duration", duration_ms, "ms")
    metrics.record("inference_success", 1.0 if success else 0.0)

def profile_inference(func):
    """Profile inference functions."""
    def wrapper(*args, **kwargs):
        with PerformanceProfiler(f"inference_{func.__name__}", metrics):
            return func(*args, **kwargs)
    return wrapper
```

### 3. Prometheus Integration

```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aimo3'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 4. Grafana Dashboard

Create a Grafana dashboard with:
- Request latency (p50, p95, p99)
- Cache hit rate
- Error rate
- GPU memory usage
- Model inference throughput
- System health status

---

## Security Hardening

### 1. Secrets Management

```bash
# Use environment variables (.env file)
KAGGLE_API_KEY=xxx
KAGGLE_USERNAME=xxx
REDIS_PASSWORD=xxx

# Or use secrets management system
# AWS Secrets Manager
# Azure Key Vault
# HashiCorp Vault
# 1Password/Bitwarden
```

### 2. API Authentication

```python
# src/api.py
from fastapi import FastAPI, Depends, HTTPException
from src.security import CredentialManager

app = FastAPI()
cred_mgr = CredentialManager()

async def verify_api_key(api_key: str):
    """Verify API key."""
    valid_key = cred_mgr.get_credential("API_KEY", required=False)
    if api_key != valid_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.post("/solve")
async def solve_problem(
    problem: str,
    api_key: str = Depends(verify_api_key)
):
    """Solve a problem (requires valid API key)."""
    from src.pipeline import AIMO3Pipeline
    pipeline = AIMO3Pipeline()
    return pipeline.solve_single_problem(problem)
```

### 3. Rate Limiting

```python
from src.security import RateLimiter
from fastapi import Request

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting."""
    if not rate_limiter.allow_request():
        return {"error": "Rate limit exceeded"}, 429
    
    response = await call_next(request)
    return response
```

### 4. Input Validation

```python
from src.security import InputValidator

@app.post("/solve")
async def solve_problem(problem: str):
    """Solve a problem with validated input."""
    # Validate input
    problem = InputValidator.validate_string(
        problem,
        min_length=10,
        max_length=10000
    )
    
    # Process...
```

### 5. Audit Logging

```python
from src.security import AuditLogger

audit = AuditLogger()

@app.post("/solve")
async def solve_problem(problem: str, user: str):
    """Solve problem with audit logging."""
    try:
        result = process_problem(problem)
        audit.log_event(
            event_type="PROBLEM_SOLVED",
            user=user,
            action="solve",
            resource="problem",
            result="success"
        )
        return result
    except Exception as e:
        audit.log_event(
            event_type="PROBLEM_FAILED",
            user=user,
            action="solve",
            resource="problem",
            result="failed",
            details={"error": str(e)}
        )
        raise
```

---

## Performance Optimization

### 1. Model Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    device_map="auto"
)

model = AutoModelForCausalLM.from_pretrained(
    "Open-Orca/orca_mini_3b",
    quantization_config=quantization_config
)
```

### 2. Batch Processing Optimization

```python
from src.pipeline import AIMO3Pipeline

pipeline = AIMO3Pipeline()

# Process in batches for better throughput
problems = load_problems()
batch_size = 32

for i in range(0, len(problems), batch_size):
    batch = problems[i:i+batch_size]
    results = pipeline.solve_batch(batch)
    save_results(results)
```

### 3. Caching Strategy

```python
from src.caching import InMemoryCache, cached

cache = InMemoryCache(max_size=10000, default_ttl_seconds=3600)

@cached(cache=cache, ttl_seconds=3600)
def solve_problem(problem_text: str):
    """Solve with caching (1 hour TTL)."""
    # Expensive computation
    return result
```

### 4. Connection Pooling

```python
# For external services
import aiohttp

class APIClient:
    def __init__(self, max_connections=100):
        self.connector = aiohttp.TCPConnector(limit=max_connections)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(connector=self.connector)
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of Memory | Model too large | Use quantization or smaller model |
| Slow inference | GPU not used | Check CUDA installation |
| Cache misses | TTL too short | Increase TTL or cache size |
| High latency | No load balancing | Deploy multiple instances |
| Model not found | Wrong model name | Verify HuggingFace model ID |

### Debugging

```bash
# Check system resources
nvidia-smi  # GPU status
free -h     # Memory
df -h       # Disk

# Check logs
tail -f logs/aimo3_*.log

# Test connectivity
curl -X GET http://localhost:8000/health

# Profile code
python -m cProfile -s cumulative src/pipeline.py

# Memory profiling
python -m memory_profiler src/pipeline.py
```

---

## Rollback Procedures

### Docker

```bash
# Tag version
docker tag aimo3-solver:latest aimo3-solver:v1.0.0

# If issue detected, rollback
docker run aimo3-solver:previous-stable-version
```

### Kubernetes

```bash
# Rollback to previous revision
kubectl rollout history deployment/aimo3-solver -n aimo3
kubectl rollout undo deployment/aimo3-solver -n aimo3 --to-revision=1
```

### Database/Cache

```bash
# Redis backup/restore
redis-cli BGSAVE
redis-cli --rdb=/path/to/backup/dump.rdb

# Restore from backup
cp /path/to/backup/dump.rdb /var/lib/redis/
redis-server /etc/redis/redis.conf
```

---

## Maintenance

### Regular Tasks

```bash
# Daily
- Check health endpoints
- Review error logs
- Verify backups completed

# Weekly
- Run security scans
- Check resource utilization
- Review performance metrics

# Monthly
- Update dependencies
- Audit access logs
- Review capacity planning
```

### Updates

```bash
# Test in staging first
docker build -t aimo3-solver:staging .
docker-compose -f docker-compose.staging.yml up

# After validation, deploy to production
# Use blue-green deployment for zero downtime
```

---

## References

- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

