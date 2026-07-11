# AWS, ap-south-1 (Mumbai) — reflects what's actually built (Phases 1-8):
# backend + speech-gateway as separate EKS deployments, managed Postgres
# (checkpointer) and Redis (rate-limit/session-state upgrade path noted in
# security/rate_limit.py). NOT applied/validated in this session — no cloud
# credentials or `terraform` CLI available here; written against the real
# architecture, not smoke-tested against a real account.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # backend "s3" {
  #   bucket = "maav-terraform-state"   # create out-of-band before first use
  #   key    = "maav/terraform.tfstate"
  #   region = "ap-south-1"
  # }
}

variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "environment" {
  type    = string
  default = "staging"
}

provider "aws" {
  region = var.region
}

# --- EKS: one cluster, two node groups (backend is CPU-bound per the load
# test; speech-gateway is connection-count-bound — see docs/LOAD_TEST_REPORT.md) ---
resource "aws_eks_cluster" "main" {
  name     = "maav-${var.environment}"
  role_arn = aws_iam_role.eks_cluster.arn
  vpc_config {
    subnet_ids = var.private_subnet_ids
  }
}

resource "aws_eks_node_group" "backend" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "backend"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["c6i.xlarge"] # CPU-optimized — backend's observed bottleneck
  scaling_config {
    min_size     = 2
    max_size     = 20
    desired_size = 2
  }
}

resource "aws_eks_node_group" "speech_gateway" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "speech-gateway"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = ["m6i.large"] # connection-count bound, not CPU-bound — memory/network matter more
  scaling_config {
    min_size     = 3
    max_size     = 50
    desired_size = 3
  }
}

# --- Managed Postgres: LangGraph checkpointer backend ---
resource "aws_db_instance" "checkpointer" {
  identifier             = "maav-checkpointer-${var.environment}"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = "db.t4g.medium"
  allocated_storage      = 50
  db_name                = "maav"
  username               = "maav"
  manage_master_user_password = true
  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  backup_retention_period = 7
  multi_az               = var.environment == "production"
}

# --- Managed Redis: rate-limiter/session-state upgrade path from
# security/rate_limit.py's in-memory-per-replica limitation ---
resource "aws_elasticache_replication_group" "session_store" {
  replication_group_id = "maav-session-${var.environment}"
  description           = "Rate-limit counters + session state, shared across gateway replicas"
  engine                = "redis"
  engine_version        = "7.1"
  node_type             = "cache.t4g.small"
  num_cache_clusters    = var.environment == "production" ? 2 : 1
  automatic_failover_enabled = var.environment == "production"
}

variable "private_subnet_ids" {
  type    = list(string)
  default = []
}

resource "aws_security_group" "db" {
  name   = "maav-db-${var.environment}"
  vpc_id = var.vpc_id
}

variable "vpc_id" {
  type    = string
  default = ""
}

resource "aws_db_subnet_group" "main" {
  name       = "maav-${var.environment}"
  subnet_ids = var.private_subnet_ids
}

resource "aws_iam_role" "eks_cluster" {
  name = "maav-eks-cluster-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role" "eks_node" {
  name = "maav-eks-node-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}
