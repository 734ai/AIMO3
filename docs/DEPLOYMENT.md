# Production Deployment Guide 🚀

## 📋 Overview
This guide covers the end-to-end deployment process for the AIMO3 Solver, including Docker/Kubernetes setups, verification steps, and production readiness checklists.

---

## 🏗️ Deployment Architectures

### 1. Docker Compose (Recommended)
Best for standalone deployments with full monitoring stack (Redis, Prometheus, Grafana).

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env.production
    depends_on: [redis]
  redis:
    image: redis:7-alpine
```

**Run:**
```bash
docker-compose up -d --build
```

### 2. Kubernetes
For high-availability clusters. Manifests are located in `k8s/`.

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## ✅ Production Readiness Checklist

Before going live, verify the following:

### 1. Security
- [ ] API Keys configured in `.env` (not version control)
- [ ] `DEBUG=False` in production
- [ ] TLS/SSL enabled for public endpoints
- [ ] Rate limiting enabled

### 2. Performance
- [ ] Model quantization enabled (if memory constrained)
- [ ] Caching (Redis) verified working
- [ ] Database connection pooling active

### 3. Monitoring
- [ ] Health endpoint `/health` returns 200 OK
- [ ] Prometheus metrics scraping active
- [ ] Logs flowing to aggregation system

---

## 🧪 Verification & Testing

Run the master verification script to validate the environment:

```bash
python verify_project.py
```

**Expected Output:**
```text
✅ Dependencies: OK
✅ Model Models: Found (gpt2)
✅ Configuration: Valid
✅ Project Health: 100%
```

### Manual Verification Steps
1. **Health Check**: `curl http://localhost:8000/health`
2. **Prediction**:
   ```bash
   curl -X POST http://localhost:8000/solve \
     -H "Content-Type: application/json" \
     -d '{"problem": "What is 2+2?"}'
   ```
3. **Load Test**: Run `locust -f tests/load_test.py`

---

## 🔄 Rollback Procedures

If issues arise:

1. **Docker**:
   ```bash
   docker-compose down
   docker tag aimo3-app:latest aimo3-app:failed
   docker tag aimo3-app:prev aimo3-app:latest
   docker-compose up -d
   ```

2. **Kubernetes**:
   ```bash
   kubectl rollout undo deployment/aimo3-solver
   ```
