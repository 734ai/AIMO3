# AIMO3 Solver - Quick Deployment Reference

## 🚀 30-Second Start

```bash
docker-compose up -d
curl http://localhost:8000/health
```

---

## 📋 Deployment Options

### Option 1: Docker Compose (Easiest)

```bash
# Start everything
docker-compose up -d

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
# Redis: localhost:6379

# Stop everything
docker-compose down
```

### Option 2: Kubernetes

```bash
kubectl create namespace aimo3
kubectl apply -f k8s/
kubectl port-forward -n aimo3 svc/aimo3-service 8000:8000
```

### Option 3: Manual Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required
KAGGLE_API_KEY=your_key
KAGGLE_USERNAME=your_username

# Optional (defaults provided)
REDIS_URL=redis://localhost:6379/0
DB_HOST=localhost
LOG_LEVEL=INFO
```

See `.env.production` for complete reference.

---

## ✅ Health Check

```bash
# API health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "components": {...}}
```

---

## 📊 Monitoring

### Grafana Dashboards
- URL: http://localhost:3000
- Username: admin
- Password: admin

### Prometheus Metrics
- URL: http://localhost:9090

### Available Dashboards
- Performance Overview
- API Statistics
- Cache Statistics
- System Resources

---

## 🔐 Security

### Change Default Credentials

```bash
# Edit .env
REDIS_PASSWORD=strong_password_here
GRAFANA_PASSWORD=strong_password_here
DB_PASSWORD=strong_password_here
API_KEY=your_api_key_here
```

### Enable HTTPS

```bash
# Place SSL cert in certs/
# Update nginx.conf
# Restart: docker-compose restart nginx
```

---

## 🐛 Troubleshooting

### Service won't start

```bash
# Check logs
docker-compose logs app

# Verify credentials
echo $KAGGLE_API_KEY

# Check ports
lsof -i :8000
```

### High memory usage

```bash
# Use quantization
docker-compose down
# Edit docker-compose.yml: LOAD_IN_8BIT=true
docker-compose up -d
```

### Slow inference

```bash
# Enable cache
# Already enabled by default

# Use smaller model
# Edit .env: MODEL_NAME=smaller-model

# Use GPU
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 ...
```

---

## 📈 Performance Tuning

### For High Throughput

```yaml
# docker-compose.yml
app:
  deploy:
    resources:
      limits:
        cpus: '8'        # More CPU cores
        memory: 16G      # More memory
```

### For Low Latency

```yaml
# docker-compose.yml
environment:
  - BATCH_SIZE=1       # Process one at a time
  - CACHE_TTL_SECONDS=3600  # Cache longer
```

### For GPU Acceleration

```bash
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

---

## 📡 API Examples

### Single Problem

```bash
curl -X POST http://localhost:8000/v1/solve \
  -H "Content-Type: application/json" \
  -d '{"problem": "What is 2+2?"}'
```

### Batch Problems

```bash
curl -X POST http://localhost:8000/v1/batch/solve \
  -H "Content-Type: application/json" \
  -d '{
    "problems": [
      {"problem_id": "P1", "problem": "2+2"},
      {"problem_id": "P2", "problem": "5*6"}
    ]
  }'
```

### Get Stats

```bash
curl http://localhost:8000/v1/stats
```

---

## 🔄 Updates & Upgrades

### Pull Latest Changes

```bash
git pull origin main
docker-compose build
docker-compose up -d
```

### Database Backup

```bash
docker-compose exec postgres pg_dump -U aimo3 aimo3 > backup.sql
```

### Database Restore

```bash
docker-compose exec postgres psql -U aimo3 aimo3 < backup.sql
```

---

## 🛑 Shutdown & Cleanup

### Stop Services (Keep Data)

```bash
docker-compose stop
```

### Remove Services & Volumes

```bash
docker-compose down -v
```

### Clean Docker System

```bash
docker system prune -a
```

---

## 📚 Additional Resources

- [Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Readiness Checklist](PRODUCTION_READINESS_CHECKLIST.md)
- [Architecture Overview](README_PRODUCTION.md)

---

## 🚨 Emergency Procedures

### Service Down

```bash
# Restart service
docker-compose restart app

# Force recreate
docker-compose down
docker-compose up -d
```

### Database Corrupted

```bash
# Restore from backup
docker-compose exec postgres psql -U aimo3 aimo3 < backup.sql

# Or use volume backup
docker volume ls  # Find volume
docker run --rm -v volume_name:/data -v $(pwd):/backup \
  alpine tar xzf /backup/backup.tar.gz -C /data
```

### Out of Memory

```bash
# Clear cache
docker-compose exec redis redis-cli FLUSHDB

# Increase memory limit
# Edit docker-compose.yml
# docker-compose restart app
```

---

## 📞 Support

- **Docs:** https://docs.aimo3.example.com
- **Email:** support@aimo3.example.com
- **Issues:** GitHub Issues

---

## ⏱️ Common Tasks & Time Estimates

| Task | Time |
|------|------|
| Initial setup | 5 min |
| First deploy | 10 min |
| Monitoring setup | 15 min |
| Custom configuration | 20 min |
| Scaling to K8s | 30 min |

---

## 🎯 Quick Decisions Matrix

```
Choose Docker Compose if:
- Single machine deployment
- Development/staging
- Team under 10 people
- <1000 req/day

Choose Kubernetes if:
- Multi-machine deployment
- Production/enterprise
- Team >10 people
- >1000 req/day
- Need auto-scaling
```

---

## ✅ Checklist Before Production

- [ ] Credentials configured
- [ ] HTTPS enabled
- [ ] Monitoring active
- [ ] Backups verified
- [ ] Load testing passed
- [ ] Team trained
- [ ] Runbooks reviewed
- [ ] On-call setup

---

**Ready to deploy? Start with:**
```bash
docker-compose up -d
```

**Questions? Check:**
```bash
cat PRODUCTION_DEPLOYMENT_GUIDE.md
```

---

*Last Updated: February 4, 2026*  
*Version: 1.0*  
*Status: ✅ Production Ready*
