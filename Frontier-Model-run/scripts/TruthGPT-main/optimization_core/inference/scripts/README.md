# 🛠️ Utility Scripts

Collection of utility scripts for managing the inference API.

## 📋 Available Scripts

### 1. `init.sh` - Initialization
Sets up the development/production environment.

```bash
chmod +x scripts/init.sh
./scripts/init.sh
```

**What it does:**
- Checks Python version
- Creates virtual environment
- Installs dependencies
- Creates .env file with secure defaults
- Creates necessary directories

### 2. `deploy.sh` - Deployment
Automated deployment script for Kubernetes.

```bash
chmod +x scripts/deploy.sh

# Basic deployment
./scripts/deploy.sh

# Skip build, use existing image
./scripts/deploy.sh --skip-build

# Custom namespace and tag
./scripts/deploy.sh --namespace production --tag v1.0.0
```

**Options:**
- `--skip-build` - Skip Docker build
- `--skip-push` - Skip image push
- `--skip-health-check` - Skip health check
- `--namespace NAME` - Kubernetes namespace
- `--tag TAG` - Docker image tag
- `--registry REGISTRY` - Container registry

### 3. `monitor.sh` - Real-time Monitoring
Real-time monitoring dashboard and alerting.

```bash
chmod +x scripts/monitor.sh

# Basic monitoring
./scripts/monitor.sh

# Custom interval
CHECK_INTERVAL=5 ./scripts/monitor.sh

# With email alerts
ALERT_EMAIL=admin@example.com ./scripts/monitor.sh

# With Slack alerts
SLACK_WEBHOOK=https://hooks.slack.com/... ./scripts/monitor.sh
```

**Features:**
- Real-time metrics display
- Health checks
- Alert thresholds (latency, errors, queue)
- Email/Slack notifications
- Auto-refresh dashboard

### 4. `backup.sh` - Backup
Automated backup of configurations and data.

```bash
chmod +x scripts/backup.sh

# Basic backup
./scripts/backup.sh

# Custom retention
RETENTION_DAYS=60 ./scripts/backup.sh
```

**What it backs up:**
- Configuration files
- Kubernetes manifests
- Environment variables
- Prometheus data (if accessible)

**Configuration:**
- `BACKUP_DIR` - Backup directory (default: ./backups)
- `RETENTION_DAYS` - Days to keep backups (default: 30)

### 5. `security_scanner.py` - Security Scanning
Comprehensive security scanning.

```bash
python -m inference.utils.security_scanner --url http://localhost:8080
```

**Checks:**
- Configuration security
- API endpoint security
- Dependency vulnerabilities
- Code security issues
- Missing security headers

## 🔧 Usage Examples

### Setup New Environment

```bash
# 1. Initialize
./scripts/init.sh

# 2. Update .env with your values
nano .env

# 3. Start API
source venv/bin/activate
python -m uvicorn inference.api:app --reload
```

### Deploy to Production

```bash
# 1. Build and deploy
./scripts/deploy.sh --namespace production --tag v1.0.0

# 2. Monitor deployment
./scripts/monitor.sh

# 3. Run security scan
python -m inference.utils.security_scanner
```

### Daily Operations

```bash
# Morning: Check health
./scripts/monitor.sh

# Backup
./scripts/backup.sh

# Security audit
python -m inference.utils.security_scanner
```

## 🔐 Security Considerations

1. **Secrets Management**
   - Never commit .env files
   - Use Kubernetes secrets in production
   - Rotate tokens regularly

2. **Script Permissions**
   - Set appropriate permissions: `chmod +x scripts/*.sh`
   - Review scripts before execution
   - Use in trusted environments only

3. **Backup Security**
   - Encrypt backups containing secrets
   - Secure backup storage
   - Limit access to backup files

## 📝 Adding Custom Scripts

Create your own scripts following this pattern:

```bash
#!/bin/bash
# Description of script

set -e

# Configuration
CONFIG_VAR=${CONFIG_VAR:-default}

# Functions
your_function() {
    echo "Doing something..."
}

# Main
main() {
    your_function
}

main
```

## 🐛 Troubleshooting

### Scripts not executable
```bash
chmod +x scripts/*.sh
```

### Permission denied
```bash
# Check if running as correct user
# Some scripts may need sudo for certain operations
```

### Scripts not found
```bash
# Ensure you're in the correct directory
cd agents/backend/onyx/server/features/Frontier-Model-run/scripts/TruthGPT-main/optimization_core/inference
```

---

**Version:** 1.0.0  
**Last Updated:** 2025-01-30


