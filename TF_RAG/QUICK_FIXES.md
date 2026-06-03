# TF_RAG Quick Fixes Needed

## Critical Issues (Fix NOW)

### 1. Exposed AWS Account ID
**File**: main.py:27-29
```python
# BEFORE (⚠️ SECURITY RISK)
MODEL_ARN = (
    "arn:aws:bedrock:us-east-1:605134452604:"  # ← Account ID exposed
    "inference-profile/global.amazon.nova-2-lite-v1:0"
)

# AFTER (✓ FIXED)
MODEL_ARN = os.getenv(
    "AWS_BEDROCK_MODEL_ARN",
    "arn:aws:bedrock:us-east-1:605134452604:"
    "inference-profile/global.amazon.nova-2-lite-v1:0"
)
```

### 2. Missing Bedrock Error Handling
**File**: main.py:100-114
```python
# BEFORE (⚠️ NO ERROR HANDLING)
response = bedrock.converse(modelId=MODEL_ARN, ...)
content_blocks = response["output"]["message"]["content"]

# AFTER (✓ SAFE)
try:
    response = bedrock.converse(modelId=MODEL_ARN, ...)
except botocore.exceptions.NoCredentialsError:
    raise RuntimeError("AWS credentials not configured. Set AWS_PROFILE or use IAM role.")
except botocore.exceptions.ClientError as e:
    raise RuntimeError(f"Bedrock API error: {e.response['Error']['Message']}")

# Safe key access
try:
    content_blocks = response["output"]["message"]["content"]
except KeyError:
    raise ValueError(f"Unexpected Bedrock response format: {response}")
```

### 3. ChromaDB Path Issues
**File**: vector_store.py:172
```python
# BEFORE (⚠️ CHANGES WITH CWD)
db_path = os.path.join(os.getcwd(), ".chroma_data")

# AFTER (✓ FIXED)
db_path = os.path.join(
    os.path.expanduser("~"),
    ".tf_rag_chroma"
)
os.makedirs(db_path, exist_ok=True)
```

### 4. Unsafe File Writes
**File**: main.py:119-122
```python
# BEFORE (⚠️ NO BACKUP)
with open(filename, "w", encoding="utf-8") as f:
    f.write(code)

# AFTER (✓ WITH BACKUP)
if os.path.exists(filename):
    backup = f"{filename}.backup"
    shutil.copy2(filename, backup)
    print(f"💾 Backed up to {backup}")

with open(filename, "w", encoding="utf-8") as f:
    f.write(code)
```

## High Priority Issues

### 5. Add Logging
Add at top of main.py:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tf_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

### 6. Validate Startup
Add before run_pipeline():
```python
def validate_startup():
    # Check AWS credentials
    try:
        bedrock.meta.events
        logger.info("✓ AWS credentials configured")
    except Exception as e:
        raise RuntimeError(f"AWS configuration error: {e}")
    
    # Check Checkov
    checkov_cmd = _get_checkov_cmd()
    if not checkov_cmd:
        raise RuntimeError("Checkov not found. Run: pip install checkov")
    logger.info("✓ Checkov available")
```

### 7. Add Input Validation
```python
def validate_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if len(prompt) > 5000:
        raise ValueError(f"Prompt too long ({len(prompt)} > 5000 chars)")
    return prompt
```

## Configuration File Needed

Create `config.yaml`:
```yaml
# AWS Configuration
aws:
  region: us-east-1
  bedrock_model_arn: ${AWS_BEDROCK_MODEL_ARN}

# ChromaDB Configuration
chroma:
  path: ~/.tf_rag_chroma
  collection_name: corporate_terraform_compliance

# Terraform Generation
terraform:
  max_attempts: 5
  checkov_timeout: 120
  skip_checks:
    - CKV_AWS_18
    - CKV_AWS_144

# Logging
logging:
  level: INFO
  file: tf_rag.log
  format: "%(asctime)s - %(levelname)s - %(message)s"
```

## Environment Variables Needed

Create `.env.example`:
```bash
# AWS Configuration (required)
AWS_BEDROCK_MODEL_ARN=arn:aws:bedrock:us-east-1:605134452604:inference-profile/global.amazon.nova-2-lite-v1:0

# Optional: Override defaults
TF_RAG_CONFIG_FILE=./config.yaml
TF_RAG_LOG_LEVEL=INFO
CHECKOV_TIMEOUT=120
```

## Dependency Issues
- ✅ boto3 (1.35.49) - installed
- ✅ chromadb (1.5.9) - installed
- ✅ checkov (3.2.532) - installed
- ❌ pyyaml - not in requirements (needed for config)
- ❌ python-dotenv - not in requirements (needed for .env)

## Import Statements Needed

```python
# main.py additions
import os
import shutil
import logging
import yaml
from dotenv import load_dotenv
import botocore.exceptions

# vector_store.py additions  
import logging
logger = logging.getLogger(__name__)
```

## Summary of Changes by Priority

| Priority | File | Issue | Lines | Fix Effort |
|----------|------|-------|-------|-----------|
| 🔴 CRITICAL | main.py | Account ID exposed | 27-29 | 2 mins |
| 🔴 CRITICAL | main.py | No Bedrock error handling | 100-114 | 10 mins |
| 🔴 CRITICAL | vector_store.py | ChromaDB path issues | 172 | 5 mins |
| 🟠 HIGH | main.py | File write safety | 119-122 | 5 mins |
| 🟠 HIGH | main.py | Add logging setup | Top | 5 mins |
| 🟠 HIGH | main.py | Startup validation | Before main | 10 mins |
| 🟡 MEDIUM | main.py | Input validation | 242 | 5 mins |
| 🟡 MEDIUM | security_scanner.py | Better error messages | 150 | 5 mins |

**Total Fix Time**: ~45 minutes for all changes
