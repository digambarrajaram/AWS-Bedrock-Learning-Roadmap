# TF_RAG: Terraform Generation with RAG & Security Scanning

A robust pipeline that uses Amazon Nova (via Bedrock) and Retrieval-Augmented Generation (RAG) to generate secure Terraform Infrastructure-as-Code, automatically scanning and fixing security violations with Checkov.

## Features

✅ **RAG-Enhanced Code Generation** - Uses ChromaDB to retrieve corporate compliance rules  
✅ **Automated Security Scanning** - Integrates Checkov for infrastructure security validation  
✅ **Self-Correcting Pipeline** - Automatically fixes violations by feeding them back to the model  
✅ **Comprehensive Error Handling** - Robust error management with detailed logging  
✅ **Production-Ready Configuration** - Environment variables, config files, and startup validation  

## Architecture

```
User Input → Validation → RAG Retrieval → Bedrock (Amazon Nova) 
  → Checkov Scan → Violations Detected → Fix Loop (max 5 attempts) 
  → Production File Output
```

## Installation

### Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- AWS Credentials configured (IAM role or AWS CLI credentials)

### Setup

1. **Clone and navigate to the TF_RAG directory:**
   ```bash
   cd TF_RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your AWS configuration
   ```

4. **Verify AWS credentials:**
   ```bash
   aws sts get-caller-identity
   ```

## Configuration

### Environment Variables

All configuration can be overridden using environment variables:

```bash
# AWS Configuration
export AWS_REGION=us-east-1
export AWS_BEDROCK_MODEL_ARN=arn:aws:bedrock:...

# Application Configuration
export TF_RAG_MAX_ATTEMPTS=5
export TF_RAG_LOG_LEVEL=INFO
export CHECKOV_TIMEOUT=120
```

Or use `.env` file:
```bash
AWS_REGION=us-east-1
AWS_BEDROCK_MODEL_ARN=arn:aws:bedrock:...
TF_RAG_MAX_ATTEMPTS=5
```

### Config File (config.yaml)

The `config.yaml` file contains detailed configuration:
- AWS region and Bedrock model
- ChromaDB vector store settings
- Terraform generation parameters
- Checkov scan configuration
- Logging settings

## Usage

### Basic Usage

```bash
python main.py
```

Then describe your infrastructure when prompted:
```
Describe your infrastructure: create a VPC with 3 public subnets and 3 private subnets across 3 availability zones
```

### Output Files

- **production_main.tf** - Successfully generated Terraform code (passes all Checkov checks)
- **review_main.tf** - Best-effort code if max attempts reached (may have violations)
- **tf_rag.log** - Detailed execution logs

## How It Works

### Pipeline Steps

1. **Input Validation**
   - Checks prompt is not empty
   - Validates prompt length (max 5000 chars)
   - Detects prompt injection patterns

2. **RAG Retrieval**
   - Queries ChromaDB for relevant compliance rules
   - Uses semantic search to find best matching rules
   - Falls back gracefully if no matches found

3. **Baseline Generation**
   - Sends prompt to Amazon Nova via Bedrock
   - Includes corporate compliance rules in system prompt
   - Extracts and cleans Terraform code

4. **Security Scanning**
   - Runs Checkov against generated code
   - Identifies security violations
   - Reports passed and failed checks

5. **Fix Loop** (up to 5 attempts)
   - If violations found, sends them back to model
   - Model generates corrected code
   - Rescans with Checkov
   - Repeats until all checks pass or max attempts reached

6. **Output**
   - Saves passing code to `production_main.tf`
   - Creates backup of previous version
   - Logs full execution details

## Error Handling

### Handled Scenarios

- ✅ AWS credentials not configured → Clear error with fix instructions
- ✅ Bedrock API rate limiting → Retryable error message
- ✅ Checkov not installed → Instructions to install
- ✅ ChromaDB initialization fails → Fallback to proceed without RAG
- ✅ Invalid Bedrock response → Detailed error logging
- ✅ File write permissions issues → Error with context
- ✅ Checkov timeout → User-friendly timeout message

### Logging

Logs are written to `tf_rag.log` with:
- Timestamp of each event
- Log level (DEBUG, INFO, WARNING, ERROR)
- Component name (main, vector_store, security_scanner)
- Full exception traces for errors

View logs:
```bash
tail -f tf_rag.log
```

## Security Features

- ✅ AWS account ID moved to environment variables (not hardcoded)
- ✅ Prompt injection prevention with pattern detection
- ✅ Response validation before parsing
- ✅ Temporary files cleaned up in finally block
- ✅ File backup created before overwriting
- ✅ Sensitive data not logged

## Troubleshooting

### Issue: "AWS credentials not found"

**Solution:** Configure AWS credentials
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Option 3: Named profile
export AWS_PROFILE=your_profile
```

### Issue: "Checkov is not available"

**Solution:** Install Checkov
```bash
pip install checkov
```

### Issue: "ChromaDB initialization failed"

**Solution:** Check disk space and permissions
```bash
# Verify home directory is writable
ls -la ~/.tf_rag_chroma/
```

### Issue: "No Bedrock response"

**Solution:** Check CloudWatch logs and ensure:
1. Bedrock is available in your region
2. Your IAM role has bedrock:InvokeModel permission
3. Model ARN is correct

## Development

### Running Tests

```bash
# Run syntax checks
python -m py_compile main.py vector_store.py security_scanner.py

# Check imports
python -c "import main; import vector_store; import security_scanner"
```

### Adding New Terraform Patterns

Edit `TERRAFORM_DOCS` in `vector_store.py`:

```python
{
    "id": "custom-pattern",
    "text": """Your Terraform Standard:
    Detailed documentation here...
    """
}
```

Then clear the ChromaDB to reseed:
```bash
rm -rf ~/.tf_rag_chroma/
```

## Performance

- **First run:** ~5-10 seconds (downloads embeddings model)
- **Typical run:** ~30-60 seconds per attempt
- **Max pipeline time:** ~3-5 minutes (5 attempts with 1 min per Bedrock call)

## Cost Estimation

Using Amazon Nova (estimated):
- Baseline generation: ~$0.001-0.005 per call
- Fix loop iterations: ~$0.001-0.005 per call
- Typical run cost: $0.01-0.05

(Prices vary by model and usage. Check AWS Bedrock pricing for current rates.)

## Files Overview

| File | Purpose |
|------|---------|
| `main.py` | Main pipeline orchestration |
| `vector_store.py` | ChromaDB vector store management |
| `security_scanner.py` | Checkov integration and scanning |
| `config.yaml` | Configuration file |
| `.env.example` | Environment variables template |
| `requirements.txt` | Python dependencies |
| `tf_rag.log` | Execution logs |
| `production_main.tf` | Generated Terraform (output) |

## Issues Fixed in This Release

### Critical (7 Issues)
1. ✅ AWS account ID exposed in code → Moved to environment variables
2. ✅ No Bedrock error handling → Comprehensive try-catch with specific errors
3. ✅ Unsafe dictionary access → Validated response structure with .get()
4. ✅ ChromaDB path issues → Fixed to use home directory consistently
5. ✅ No AWS credentials validation → Added startup validation
6. ✅ Unsafe file writes → Added backup creation
7. ✅ Empty response edge case → Added validation at return

### High Priority (5 Issues)
8. ✅ No logging system → Added Python logging with file output
9. ✅ Hardcoded configuration → Moved to config.yaml and environment variables
10. ✅ File write without safety → Added permission checks and backups
11. ✅ Input validation missing → Added length and injection prevention
12. ✅ Poor error messages → Detailed error context and recovery instructions

## Contributing

To report issues or suggest improvements:

1. Review `ISSUES_AND_ERRORS_ANALYSIS.md` for all identified issues
2. Check `tf_rag.log` for detailed error information
3. Run syntax checks: `python -m py_compile main.py`
4. Test with sample inputs before committing

## License

[Add your license information here]

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review execution logs: `tail -f tf_rag.log`
3. Run with debug logging: `TF_RAG_LOG_LEVEL=DEBUG python main.py`

---

**Version**: 2.0 (Fixed Release)  
**Last Updated**: June 3, 2026  
**Status**: Production Ready ✅
