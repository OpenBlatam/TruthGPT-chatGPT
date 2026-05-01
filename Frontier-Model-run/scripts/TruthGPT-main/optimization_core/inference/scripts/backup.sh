#!/bin/bash
# 💾 Backup Script
# Automated backup for inference API data and configurations

set -e

# Configuration
BACKUP_DIR=${BACKUP_DIR:-./backups}
RETENTION_DAYS=${RETENTION_DAYS:-30}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="inference-api-backup-${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

create_backup() {
    echo -e "${YELLOW}Creating backup...${NC}"
    
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
    
    # Backup configuration files
    echo "Backing up configurations..."
    if [ -d "configs" ]; then
        cp -r configs "${BACKUP_DIR}/${BACKUP_NAME}/"
    fi
    
    # Backup Kubernetes manifests
    if [ -d "k8s" ]; then
        cp -r k8s "${BACKUP_DIR}/${BACKUP_NAME}/"
    fi
    
    # Backup environment variables
    echo "Backing up environment..."
    env | grep -E "^(TRUTHGPT|REDIS|CACHE|BATCH|RATE|CIRCUIT|ENABLE|OTLP|WEBHOOK)" > "${BACKUP_DIR}/${BACKUP_NAME}/env.txt" || true
    
    # Backup Prometheus data (if accessible)
    if kubectl get namespace prometheus >/dev/null 2>&1; then
        echo "Backing up Prometheus data..."
        kubectl exec -n prometheus prometheus-0 -- tar czf - /prometheus 2>/dev/null | \
            cat > "${BACKUP_DIR}/${BACKUP_NAME}/prometheus-data.tar.gz" || true
    fi
    
    # Create archive
    echo "Creating archive..."
    cd "${BACKUP_DIR}"
    tar czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
    rm -rf "${BACKUP_NAME}"
    
    echo -e "${GREEN}✓ Backup created: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz${NC}"
}

cleanup_old_backups() {
    echo -e "${YELLOW}Cleaning up old backups (retention: ${RETENTION_DAYS} days)...${NC}"
    find "${BACKUP_DIR}" -name "*.tar.gz" -mtime +${RETENTION_DAYS} -delete
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

main() {
    create_backup
    cleanup_old_backups
    echo -e "${GREEN}🎉 Backup complete!${NC}"
}

main


