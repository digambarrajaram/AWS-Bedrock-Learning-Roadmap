# TF_RAG Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
cd TF_RAG
pip install -r requirements.txt
```

### 2. Configure AWS
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Or set AWS_PROFILE if using named profile
export AWS_PROFILE=your-profile
```

### 3. Run the Pipeline
```bash
python main.py
```

When prompted, describe your infrastructure:
```
Describe your infrastructure: Create a VPC with 3 public and 3 private subnets across 3 AZs
```

### 4. Check Output
- **Success**: Look for `production_main.tf` with your generated Terraform code
- **Logs**: Check `tf_rag.log` for detailed execution information
- **Backup**: Previous version saved as `production_main.tf.backup`

---

## Configuration (Optional)

### Method 1: Environment Variables
```bash
export AWS_REGION=us-west-2
export AWS_BEDROCK_MODEL_ARN=arn:aws:bedrock:...
export TF_RAG_MAX_ATTEMPTS=3
export TF_RAG_LOG_LEVEL=DEBUG
python main.py
```

### Method 2: .env File
```bash
cp .env.example .env
# Edit .env with your settings
python main.py  # Automatically loads .env
```

### Method 3: config.yaml
```bash
# Already configured with sensible defaults
# Edit config.yaml to customize
```

---

## Troubleshooting

### ❌ "AWS credentials not found"
```bash
# Fix 1: Configure AWS CLI
aws configure

# Fix 2: Set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Fix 3: Use IAM role (on EC2/Lambda)
# Automatically picked up
```

### ❌ "Bedrock model not found"
```bash
# Verify Bedrock availability in your region
aws bedrock list-foundation-models --region us-east-1 | head -20

# Check model ARN is correct
export AWS_BEDROCK_MODEL_ARN=arn:aws:bedrock:us-east-1:605134452604:inference-profile/global.amazon.nova-2-lite-v1:0
```

### ❌ "Checkov is not available"
```bash
pip install checkov
# Or reinstall
pip install --force-reinstall checkov
```

### ❌ "ChromaDB initialization failed"
```bash
# Check home directory is writable
ls -la ~/.tf_rag_chroma/

# Or delete and recreate
rm -rf ~/.tf_rag_chroma/
python main.py  # Will recreate
```

### ❌ "Check execution logs"
```bash
tail -f tf_rag.log
# For more details, set debug logging:
export TF_RAG_LOG_LEVEL=DEBUG
python main.py
```

---

## Common Use Cases

### Generate VPC
```
Describe your infrastructure: Create a VPC with 3 public subnets and 3 private subnets in 3 availability zones with NAT gateways
```

### Generate EKS Cluster
```
Describe your infrastructure: Create an EKS cluster with auto-scaling node groups using BOTTLEROCKET AMI and enable IRSA
```

### Generate S3 Bucket
```
Describe your infrastructure: Create a secure S3 bucket with versioning, encryption, lifecycle policies, and public access blocked
```

### Generate Multi-Service Infrastructure
```
Describe your infrastructure: Create a VPC with EKS cluster, RDS PostgreSQL database with multi-AZ, and S3 bucket for backups, all with appropriate security groups and IAM roles
```

---

## What Gets Generated

The pipeline generates multiple Terraform files:

```hcl
# === FILE: main.tf ===
provider "aws" {
  region = var.aws_region
  default_tags { tags = local.common_tags }
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = ">= 5.0"
  # ... VPC configuration
}

# === FILE: variables.tf ===
variable "aws_region" { type = string; default = "us-east-1" }
variable "environment" { type = string }
# ... other variables

# === FILE: outputs.tf ===
output "vpc_id" {
  description = "ID of the provisioned VPC"
  value       = module.vpc.vpc_id
}
# ... other outputs
```

---

## Output Files

| File | Purpose |
|------|---------|
| `production_main.tf` | ✅ Generated Terraform (passes all checks) |
| `production_main.tf.backup` | Previous version backup |
| `review_main.tf` | Best-effort code if max attempts reached |
| `tf_rag.log` | Detailed execution logs |

---

## Pipeline Flow

```
1. Input Validation
   ↓
2. RAG Context Retrieval (ChromaDB)
   ↓
3. Baseline Terraform Generation (Amazon Nova)
   ↓
4. Checkov Security Scan
   ↓
   ├─ ✅ Pass? → Save to production_main.tf [SUCCESS]
   │
   ├─ ❌ Fail? → Send violations back to model
   │   ↓
   │   └─ Up to 5 attempts to fix
   │       ├─ ✅ Pass? → Save [SUCCESS]
   │       └─ ❌ Fail? → Save to review_main.tf [MAX ATTEMPTS]
```

---

## Performance Tips

### Speed Up Repeated Runs
- ChromaDB embeddings are cached in `~/.tf_rag_chroma/`
- Subsequent runs reuse cached embeddings
- First run takes longer as it downloads the embedding model

### Reduce API Costs
- Use smaller/simpler prompts when possible
- CloudaDB RAG reduces need for model regeneration
- Typical cost: $0.01-0.05 per run

### Optimize for Compliance
- More specific compliance rules in `vector_store.py` → Better first-pass code
- Detailed prompts → Fewer fix iterations needed

---

## Advanced Configuration

### Change Checkov Rules
Edit `security_scanner.py` line ~100:
```python
cmd_args = checkov_cmd + [
    "-d", tmp_dir,
    "--output", "json",
    "--skip-check", "CKV_AWS_18,CKV_AWS_144"  # ← Modify here
]
```

### Add Custom Terraform Patterns
Edit `vector_store.py` in TERRAFORM_DOCS:
```python
{
    "id": "my-pattern",
    "text": """My Custom Pattern:
    Detailed Terraform standards...
    """
}
```
Then clear ChromaDB: `rm -rf ~/.tf_rag_chroma/`

### Change Max Attempts
```bash
export TF_RAG_MAX_ATTEMPTS=10
python main.py
```

---

## Debugging

### Enable Debug Logging
```bash
export TF_RAG_LOG_LEVEL=DEBUG
python main.py
# Check tf_rag.log for detailed information
```

### Test AWS Credentials
```bash
aws sts get-caller-identity
# Should return your AWS account ID and user ARN
```

### Test Checkov
```bash
echo 'resource "aws_s3_bucket" "test" {}' > test.tf
checkov -f test.tf
rm test.tf
```

### Test ChromaDB
```bash
python -c "from vector_store import setup_vector_store; c = setup_vector_store(); print(f'ChromaDB ready: {c.count()} rules')"
```

---

## Next Steps

1. **Generate Terraform Code**: Run pipeline for your infrastructure
2. **Review Code**: Check `production_main.tf` and `tf_rag.log`
3. **Customize**: Edit generated code as needed
4. **Test**: Run `terraform plan` and `terraform apply`
5. **Deploy**: Use Terraform to provision infrastructure

---

## Support

- 📖 **Full Documentation**: See `README_FIXED.md`
- 🔍 **Issue Details**: See `ISSUES_AND_ERRORS_ANALYSIS.md`
- ✅ **What Was Fixed**: See `RESOLUTION_SUMMARY.md`
- 🐛 **Debug Info**: Check `tf_rag.log` file

---

## Files Reference

```
TF_RAG/
├── main.py                           # Main pipeline (execute this)
├── vector_store.py                  # RAG context retrieval
├── security_scanner.py              # Checkov integration
├── config.yaml                      # Configuration file
├── .env.example                     # Environment variables template
├── requirements.txt                 # Python dependencies
├── production_main.tf               # OUTPUT: Generated Terraform
├── tf_rag.log                       # OUTPUT: Execution logs
├── README_FIXED.md                  # Full documentation
├── RESOLUTION_SUMMARY.md            # What was fixed
├── ISSUES_AND_ERRORS_ANALYSIS.md    # Detailed issues
├── QUICK_FIXES.md                   # Code examples
└── QUICK_START.md                   # This file
```

---

**Ready to generate Terraform?** Run: `python main.py`

✅ **All systems ready. Happy Terraforming!** 🚀
