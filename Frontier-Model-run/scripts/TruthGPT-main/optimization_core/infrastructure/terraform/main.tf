# TruthGPT Infrastructure - Terraform Configuration
# =================================================
# Complete infrastructure setup for TruthGPT optimization framework

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "azurerm" {
    resource_group_name  = "truthgpt-tfstate"
    storage_account_name = "truthgpttfstate"
    container_name       = "tfstate"
    key                  = "truthgpt.tfstate"
  }
}

# Configure the Azure Provider
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults  = true
    }
  }
}

# Configure the Kubernetes Provider
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.cluster_ca_certificate)
}

# Configure the Helm Provider
provider "helm" {
  kubernetes {
    host                   = azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.host
    client_certificate     = base64decode(azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.client_certificate)
    client_key             = base64decode(azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.client_key)
    cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.truthgpt_aks.kube_config.0.cluster_ca_certificate)
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "East US"
}

variable "cluster_name" {
  description = "AKS cluster name"
  type        = string
  default     = "truthgpt-aks"
}

variable "node_count" {
  description = "Number of nodes in the cluster"
  type        = number
  default     = 3
}

variable "node_vm_size" {
  description = "VM size for cluster nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "gpu_node_count" {
  description = "Number of GPU nodes"
  type        = number
  default     = 2
}

variable "gpu_vm_size" {
  description = "VM size for GPU nodes"
  type        = string
  default     = "Standard_NC6s_v3"
}

# Data sources
data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "truthgpt_rg" {
  name     = "truthgpt-${var.environment}-rg"
  location = var.location
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
    Team        = "ML-Engineering"
    CostCenter  = "AI-Research"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "truthgpt_vnet" {
  name                = "truthgpt-${var.environment}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.truthgpt_rg.location
  resource_group_name = azurerm_resource_group.truthgpt_rg.name
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# Subnet for AKS
resource "azurerm_subnet" "truthgpt_subnet" {
  name                 = "truthgpt-${var.environment}-subnet"
  resource_group_name  = azurerm_resource_group.truthgpt_rg.name
  virtual_network_name = azurerm_virtual_network.truthgpt_vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Subnet for GPU nodes
resource "azurerm_subnet" "truthgpt_gpu_subnet" {
  name                 = "truthgpt-${var.environment}-gpu-subnet"
  resource_group_name  = azurerm_resource_group.truthgpt_rg.name
  virtual_network_name = azurerm_virtual_network.truthgpt_vnet.name
  address_prefixes     = ["10.0.2.0/24"]
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "truthgpt_logs" {
  name                = "truthgpt-${var.environment}-logs"
  location            = azurerm_resource_group.truthgpt_rg.location
  resource_group_name = azurerm_resource_group.truthgpt_rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# Application Insights
resource "azurerm_application_insights" "truthgpt_insights" {
  name                = "truthgpt-${var.environment}-insights"
  location            = azurerm_resource_group.truthgpt_rg.location
  resource_group_name = azurerm_resource_group.truthgpt_rg.name
  workspace_id        = azurerm_log_analytics_workspace.truthgpt_logs.id
  application_type    = "web"
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# Key Vault
resource "azurerm_key_vault" "truthgpt_kv" {
  name                = "truthgpt-${var.environment}-kv"
  location            = azurerm_resource_group.truthgpt_rg.location
  resource_group_name = azurerm_resource_group.truthgpt_rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  
  purge_protection_enabled = true
  
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id
    
    key_permissions = [
      "Get", "List", "Create", "Delete", "Update", "Import", "Backup", "Restore", "Recover"
    ]
    
    secret_permissions = [
      "Get", "List", "Set", "Delete", "Backup", "Restore", "Recover"
    ]
    
    certificate_permissions = [
      "Get", "List", "Create", "Delete", "Update", "Import", "Backup", "Restore", "Recover"
    ]
  }
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# Container Registry
resource "azurerm_container_registry" "truthgpt_acr" {
  name                = "truthgpt${var.environment}acr"
  resource_group_name = azurerm_resource_group.truthgpt_rg.name
  location            = azurerm_resource_group.truthgpt_rg.location
  sku                 = "Premium"
  admin_enabled       = true
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "truthgpt_aks" {
  name                = var.cluster_name
  location            = azurerm_resource_group.truthgpt_rg.location
  resource_group_name = azurerm_resource_group.truthgpt_rg.name
  dns_prefix          = "truthgpt-${var.environment}"
  kubernetes_version  = "1.28"
  
  default_node_pool {
    name                = "system"
    node_count          = var.node_count
    vm_size             = var.node_vm_size
    vnet_subnet_id      = azurerm_subnet.truthgpt_subnet.id
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 10
    
    node_taints = [
      "CriticalAddonsOnly=true:NoSchedule"
    ]
  }
  
  # GPU node pool
  node_pool {
    name                = "gpu"
    node_count          = var.gpu_node_count
    vm_size             = var.gpu_vm_size
    vnet_subnet_id      = azurerm_subnet.truthgpt_gpu_subnet.id
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 20
    
    node_taints = [
      "nvidia.com/gpu=true:NoSchedule"
    ]
    
    node_labels = {
      "node-type" = "gpu-enabled"
    }
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.truthgpt_logs.id
  }
  
  azure_policy_enabled = true
  
  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
    service_cidr     = "10.1.0.0/16"
    dns_service_ip   = "10.1.0.10"
  }
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# Attach ACR to AKS
resource "azurerm_role_assignment" "truthgpt_acr_pull" {
  scope                = azurerm_container_registry.truthgpt_acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.truthgpt_aks.kubelet_identity[0].object_id
}

# Storage Account for persistent volumes
resource "azurerm_storage_account" "truthgpt_storage" {
  name                     = "truthgpt${var.environment}storage"
  resource_group_name      = azurerm_resource_group.truthgpt_rg.name
  location                 = azurerm_resource_group.truthgpt_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# File Share for persistent volumes
resource "azurerm_storage_share" "truthgpt_share" {
  name                 = "truthgpt-${var.environment}-share"
  storage_account_name = azurerm_storage_account.truthgpt_storage.name
  quota                = 100
}

# Redis Cache
resource "azurerm_redis_cache" "truthgpt_redis" {
  name                = "truthgpt-${var.environment}-redis"
  location            = azurerm_resource_group.truthgpt_rg.location
  resource_group_name = azurerm_resource_group.truthgpt_rg.name
  capacity            = 1
  family              = "C"
  sku_name            = "Standard"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  tags = {
    Environment = var.environment
    Project     = "TruthGPT"
  }
}

# Outputs
output "resource_group_name" {
  value = azurerm_resource_group.truthgpt_rg.name
}

output "cluster_name" {
  value = azurerm_kubernetes_cluster.truthgpt_aks.name
}

output "cluster_fqdn" {
  value = azurerm_kubernetes_cluster.truthgpt_aks.fqdn
}

output "cluster_identity" {
  value = azurerm_kubernetes_cluster.truthgpt_aks.identity
}

output "acr_login_server" {
  value = azurerm_container_registry.truthgpt_acr.login_server
}

output "key_vault_name" {
  value = azurerm_key_vault.truthgpt_kv.name
}

output "log_analytics_workspace_id" {
  value = azurerm_log_analytics_workspace.truthgpt_logs.id
}

output "application_insights_connection_string" {
  value = azurerm_application_insights.truthgpt_insights.connection_string
  sensitive = true
}

output "redis_hostname" {
  value = azurerm_redis_cache.truthgpt_redis.hostname
}

output "storage_account_name" {
  value = azurerm_storage_account.truthgpt_storage.name
}

output "storage_share_name" {
  value = azurerm_storage_share.truthgpt_share.name
}


