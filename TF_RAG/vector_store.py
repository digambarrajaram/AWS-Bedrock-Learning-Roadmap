import os
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Corporate compliance documentation seeded into the vector store.
# Each entry represents a distinct Terraform pattern or standard that the
# LLM should follow when generating infrastructure code.
# ---------------------------------------------------------------------------
TERRAFORM_DOCS = [
    # ── Provider & backend ──────────────────────────────────────────────────
    {
        "id": "provider-aws",
        "text": """AWS Provider Standard:
        Always pin the provider version. Minimum: hashicorp/aws >= 5.0.
        Set default_tags inside the provider block so every resource inherits
        mandatory tags (Environment, Team, CostCenter, ManagedBy=Terraform).
        Example:
            terraform {
              required_providers {
                aws = { source = "hashicorp/aws", version = ">= 5.0" }
              }
              required_version = ">= 1.6"
            }
            provider "aws" {
              region = var.aws_region
              default_tags { tags = local.common_tags }
            }
        Never hard-code credentials or region strings outside of variables.""",
    },
    {
        "id": "backend-s3",
        "text": """Remote State Backend Standard:
        Always configure an S3 + DynamoDB remote backend for state locking.
        Example:
            terraform {
              backend "s3" {
                bucket         = "my-company-tf-state"
                key            = "<project>/<env>/terraform.tfstate"
                region         = "us-east-1"
                dynamodb_table = "terraform-state-lock"
                encrypt        = true
              }
            }
        The S3 bucket must have versioning and SSE-S3 encryption enabled.
        Never use local backend for production workloads.""",
    },

    # ── Variables ────────────────────────────────────────────────────────────
    {
        "id": "variables-standard",
        "text": """Variables File Standard (variables.tf):
        Every module must expose input variables in a dedicated variables.tf.
        All variables need: description, type, and either a default or marked
        as required (no default).  Sensitive values must carry sensitive=true.
        Standard variables every root module must include:
            variable "aws_region"   { type = string; default = "us-east-1" }
            variable "environment"  { type = string }          # required
            variable "project_name" { type = string }          # required
            variable "team"         { type = string }          # required
            variable "cost_center"  { type = string }          # required
        Use locals.tf for derived values like common_tags:
            locals {
              common_tags = {
                Environment = var.environment
                Team        = var.team
                CostCenter  = var.cost_center
                ManagedBy   = "Terraform"
              }
            }""",
    },
    {
        "id": "tfvars-standard",
        "text": """Terraform Var Files (.tfvars):
        Always ship a terraform.tfvars.example file committed to the repo.
        Never commit terraform.tfvars containing real values to source control.
        Add terraform.tfvars to .gitignore.
        Use separate var files per environment:
            environments/
              dev.tfvars
              staging.tfvars
              prod.tfvars
        Apply with: terraform apply -var-file=environments/prod.tfvars""",
    },

    # ── Outputs ──────────────────────────────────────────────────────────────
    {
        "id": "outputs-standard",
        "text": """Outputs File Standard (outputs.tf):
        Every module must expose outputs in a dedicated outputs.tf.
        Always provide: output name, description, value, and sensitive flag
        where applicable.
        Mandatory outputs for commonly used resources:
          VPC:        vpc_id, private_subnet_ids, public_subnet_ids
          EKS:        cluster_name, cluster_endpoint, cluster_oidc_issuer_url
          S3:         bucket_id, bucket_arn, bucket_domain_name
          RDS:        db_instance_endpoint, db_instance_id (NOT password)
          ALB:        alb_arn, alb_dns_name, target_group_arn
        Example:
            output "vpc_id" {
              description = "ID of the provisioned VPC."
              value       = module.vpc.vpc_id
            }
        Mark outputs containing secrets as sensitive=true.""",
    },

    # ── EKS ──────────────────────────────────────────────────────────────────
    {
        "id": "eks-module",
        "text": """EKS Module Standard:
        Use module source 'terraform-aws-modules/eks/aws', version >= 20.0.
        Required variables: cluster_name, cluster_version (min '1.30'),
        vpc_id, subnet_ids (private subnets only for the control plane).
        Mandatory settings:
          enable_irsa                          = true
          cluster_endpoint_private_access      = true
          cluster_endpoint_public_access       = false   # for production
          enable_cluster_creator_admin_permissions = true
        Managed node groups: min_size=2, max_size=10, desired_size=2.
        Use BOTTLEROCKET_x86_64 AMI type for node groups.
        Enable aws-auth ConfigMap management via the module.
        Mandatory outputs: cluster_name, cluster_endpoint,
          cluster_certificate_authority_data, cluster_oidc_issuer_url.
        Tags: Environment, Team, CostCenter are mandatory.""",
    },

    # ── S3 ───────────────────────────────────────────────────────────────────
    {
        "id": "s3-module",
        "text": """S3 Bucket Standard:
        Use separate modern resources — never deprecated inline blocks.
          aws_s3_bucket                              – core bucket
          aws_s3_bucket_server_side_encryption_configuration – SSE-S3 (AES256)
          aws_s3_bucket_versioning                   – enabled for prod
          aws_s3_bucket_public_access_block          – all four flags = true
          aws_s3_bucket_lifecycle_configuration      – include abort_incomplete_multipart_upload
          aws_s3_bucket_policy                       – enforce HTTPS-only
        Never use ACLs (set object_ownership = BucketOwnerEnforced).
        Do NOT add SNS topics or cross-region replication unless explicitly requested.
        Mandatory outputs: bucket_id, bucket_arn, bucket_regional_domain_name.""",
    },

    # ── VPC ──────────────────────────────────────────────────────────────────
    {
        "id": "vpc-module",
        "text": """VPC Module Standard:
        Use module source 'terraform-aws-modules/vpc/aws', version >= 5.0.
        Always create 3 public + 3 private subnets spread across 3 AZs.
        Mandatory settings:
          enable_nat_gateway   = true
          single_nat_gateway   = false   # one NAT GW per AZ for production HA
          enable_dns_hostnames = true
          enable_dns_support   = true
          enable_flow_log      = true
          flow_log_destination_type                    = "cloud-watch-logs"
          flow_log_cloudwatch_log_group_retention_days = 30
          flow_log_traffic_type                        = "ALL"
        Tag public subnets with kubernetes.io/role/elb=1 for EKS ingress.
        Tag private subnets with kubernetes.io/role/internal-elb=1.
        Mandatory outputs: vpc_id, private_subnets, public_subnets,
          vpc_cidr_block, nat_public_ips.""",
    },

    # ── RDS ──────────────────────────────────────────────────────────────────
    {
        "id": "rds-standard",
        "text": """RDS Instance Standard:
        Use aws_db_instance or module 'terraform-aws-modules/rds/aws' >= 6.0.
        Mandatory settings:
          multi_az               = true    (production)
          storage_encrypted      = true
          deletion_protection    = true
          skip_final_snapshot    = false
          backup_retention_period = 7      (minimum)
          auto_minor_version_upgrade = true
          publicly_accessible    = false"""
    }
]


# ---------------------------------------------------------------------------
# Vector Store Engine Implementations
# ---------------------------------------------------------------------------

def setup_vector_store() -> chromadb.Collection:
    """
    Initializes a persistent local ChromaDB instance, creates/fetches 
    the collection, and seeds it with compliance records if empty.
    """
    # Create a local cache path for the database inside your project structure
    db_path = os.path.join(os.getcwd(), ".chroma_data")
    
    client = chromadb.PersistentClient(path=db_path)
    
    # Use standard default sentence-transformer embedding model
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name="corporate_terraform_compliance",
        embedding_function=embedding_fn
    )
    
    # Only seed data if database collection is completely fresh
    if collection.count() == 0:
        documents = [doc["text"] for doc in TERRAFORM_DOCS]
        ids = [doc["id"] for doc in TERRAFORM_DOCS]
        
        collection.add(
            documents=documents,
            ids=ids
        )
        
    return collection


def retrieve_context(collection: chromadb.Collection, query: str, n_results: int = 1) -> str:
    """
    Queries ChromaDB vector collection and drops most context-relevant string.
    """
    if not query.strip():
        return ""
        
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Combine retrieved document context records into a flat string payload
    if results and "documents" in results and results["documents"]:
        flat_docs = results["documents"][0]
        return "\n\n".join(flat_docs)
