# TF_RAG Issues and Errors Analysis

**Date**: June 3, 2026  
**Scope**: TF_RAG folder - Code quality, runtime issues, and error handling  
**Status**: ⚠️ Multiple issues identified - requires attention before production use

---

## Executive Summary

The TF_RAG pipeline has:
- ✅ **No syntax errors** (all Python files parse correctly)
- ✅ **Dependencies installed** (boto3, chromadb, checkov all present)
- ❌ **7 Critical issues** affecting runtime stability and security
- ⚠️ **10+ Code quality issues** limiting maintainability and robustness

---

## Critical Issues (Priority: HIGH)

### 1. **Missing Error Handling in Bedrock API Call** [main.py:100-110]
**Severity**: 🔴 CRITICAL  
**Issue**: `generate_terraform()` calls `bedrock.converse()` without try-catch

```python
response = bedrock.converse(
    modelId=MODEL_ARN,
    messages=messages,
    system=system_prompt,
    inferenceConfig={"maxTokens": 4096, "temperature": 0.1},
    additionalModelRequestFields={
        "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}
    },
)
```

**Risks**:
- Network timeout → unhandled exception → script crash
- Invalid credentials → botocore.exceptions.NoCredentialsError
- Rate limiting → botocore.exceptions.ClientError
- Malformed response → KeyError when accessing response["output"]["message"]["content"]

**Impact**: Pipeline fails silently with cryptic error messages  
**Fix Required**: Add try-catch with specific error handling

---

### 2. **AWS Account ID Exposed in Code** [main.py:27-29]
**Severity**: 🔴 CRITICAL - SECURITY ISSUE  
**Issue**: Account ID hardcoded in MODEL_ARN

```python
MODEL_ARN = (
    "arn:aws:bedrock:us-east-1:605134452604:"
    "inference-profile/global.amazon.nova-2-lite-v1:0"
)
```

**Risks**:
- Account ID: `605134452604` publicly exposed in source code
- Can be used for reconnaissance attacks
- If repo is pushed to GitHub, it's permanently indexed

**Impact**: Security exposure  
**Fix Required**: Move to environment variables or AWS Secrets Manager

---

### 3. **Unsafe Dictionary Access - KeyError Risk** [main.py:112-114]
**Severity**: 🔴 CRITICAL  
**Issue**: Direct key access without validation

```python
content_blocks = response["output"]["message"]["content"]
raw_code = next(
    (block["text"] for block in content_blocks if "text" in block), ""
)
```

**Risks**:
- If response structure differs: `KeyError: 'output'`
- If "content" is not a list: TypeError when iterating
- Silent failure to next empty string if no "text" block exists

**Impact**: Unhandled exceptions crash pipeline  
**Fix Required**: Use .get() method with defaults

---

### 4. **ChromaDB Path Dependency on Current Working Directory** [vector_store.py:172]
**Severity**: 🔴 CRITICAL  
**Issue**: Path hardcoded to relative directory

```python
db_path = os.path.join(os.getcwd(), ".chroma_data")
```

**Risks**:
- Running script from different directory = separate database
- CWD changes → loses access to existing embeddings
- Multiple instances create duplicate embeddings

**Impact**: Data isolation and consistency issues  
**Fix Required**: Use absolute path or config-based location

---

### 5. **No AWS Credentials Validation** [main.py:22]
**Severity**: 🔴 CRITICAL  
**Issue**: Bedrock client created without credential checks

```python
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
```

**Risks**:
- If AWS credentials not configured: NoCredentialsError only occurs at first API call
- No early failure detection
- Script runs empty prompts before failing

**Impact**: Delayed error detection, wasted execution time  
**Fix Required**: Add startup validation

---

### 6. **File Write Without Permission Checking** [main.py:119-122]
**Severity**: 🟠 HIGH  
**Issue**: Files written without validation

```python
def _save(code: str, filename: str = "production_main.tf") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
```

**Risks**:
- No check if file already exists (silent overwrite)
- No permission validation
- Relative path issues (see issue #4)
- No backup of previous version

**Impact**: Data loss, production file overwritten  
**Fix Required**: Add backups and permission checks

---

### 7. **Empty Response Handling Ambiguity** [main.py:123-142]
**Severity**: 🟠 HIGH  
**Issue**: Function returns raw_code even if empty

```python
if not raw_code:
    raise ValueError("No text output blocks found in the model response.")

code_block_match = re.search(r"(?s)```(?:hcl|terraform)?\s*(.*?)\s*```", raw_code)
return code_block_match.group(1).strip() if code_block_match else raw_code.strip()
```

**Logic Bug**:
1. If raw_code is empty → raises ValueError ✓
2. If regex finds ```...``` blocks → returns inner content ✓
3. If regex DOESN'T find blocks → returns raw_code even if it's whitespace ✓
4. BUT if raw_code contains only whitespace → returns empty string → later operations fail

**Impact**: Invalid Terraform generated, Checkov scan fails  
**Fix Required**: Validate non-empty code at return

---

## Code Quality Issues (Priority: MEDIUM)

### 8. **No Logging System**
**Issue**: Only print() statements for debugging  
**Impact**: 
- No log files for troubleshooting
- No error timestamps
- Cannot track execution history

**Recommendation**: Use Python logging module

---

### 9. **Hardcoded Configuration Values**
**Issues**:
- Region: `us-east-1` hardcoded in 3 places
- Model ARN hardcoded
- MAX_ATTEMPTS = 5 not configurable
- Checkov skip checks hardcoded: `CKV_AWS_18,CKV_AWS_144`

**Impact**: Cannot easily switch environments or adjust behavior  
**Recommendation**: Use config file or environment variables

---

### 10. **Potential Infinite Prompt Loop Risk** [main.py:145-190]
**Issue**: If model continuously generates invalid code, loop runs 5 times

```python
for attempt in range(1, MAX_ATTEMPTS + 1):
    # ... scan and fix logic ...
```

**Scenario**:
1. Checkov finds violations
2. Send violations back to model
3. Model generates similar invalid code
4. Repeat 5 times
5. Save to `review_main.tf` → requires manual cleanup

**Impact**: Wasted API calls (~5 Bedrock calls per run), poor user experience  
**Recommendation**: Add detection for repeated violations or model confusion

---

### 11. **Incomplete Error Messages in security_scanner.py**
**Issue**: Parse errors truncated [line 150]

```python
return 0, 0, [f"Could not parse Checkov output as JSON. Raw snippet: {stdout[:150]}"]
```

**Problem**: Only first 150 chars shown, full context needed for debugging  
**Recommendation**: Full output with line breaks

---

### 12. **Checkov Platform-Specific Complexity**
**Issue**: 20+ lines searching for checkov executable [security_scanner.py:35-65]

**Risks**:
- Multiple probe attempts with timeouts (slow)
- Complex fallback logic hard to maintain
- Test coverage likely missing

**Recommendation**: Simplify or delegate to package manager

---

### 13. **Missing Input Validation** [main.py:242]
**Issue**: User input not validated

```python
user_prompt = input("Describe your infrastructure: ")
if not user_prompt.strip():
    user_prompt = "create a vpc"  # Default if empty
```

**Problems**:
- No length validation (could be 10,000 chars)
- No prompt injection validation
- Default "create a vpc" might not be desired

**Recommendation**: Add length limits and sanitization

---

### 14. **No Timeout on Bedrock Calls**
**Issue**: `generate_terraform()` has no timeout

**Risk**:
- Model could hang indefinitely
- No recovery mechanism
- Script becomes unresponsive

**Recommendation**: Add explicit timeout or use boto3 timeouts

---

### 15. **Unused Variable in evaluate_history** [main.py:161]
**Issue**: `history` list not fully utilized

```python
history = [{"role": "user", "content": [{"text": f"Generate a secure baseline..."}]}]
# ... later ...
history.append({"role": "assistant", "content": [{"text": tf_code}]})
```

**Problem**: 
- First message duplicates initial prompt
- Could cause model confusion

**Recommendation**: Review conversation flow logic

---

## Runtime Issues Checklist

| Issue | Status | Severity |
|-------|--------|----------|
| Bedrock API error handling | ❌ Missing | CRITICAL |
| AWS credentials validation | ❌ Missing | CRITICAL |
| Response structure validation | ❌ Missing | CRITICAL |
| ChromaDB path isolation | ❌ Missing | CRITICAL |
| File write safety | ❌ Missing | HIGH |
| Logging and debugging | ❌ Missing | HIGH |
| Configuration management | ❌ Missing | HIGH |
| Input validation | ❌ Missing | MEDIUM |
| Checkov output handling | ⚠️ Partial | MEDIUM |
| Error messages | ⚠️ Minimal | MEDIUM |

---

## Recommended Fixes (Priority Order)

### Phase 1: Critical (Must Fix Before Production)
1. **Add Bedrock error handling**
   - Catch specific exceptions (NoCredentialsError, ClientError, etc.)
   - Provide actionable error messages
   - Implement retry logic for transient failures

2. **Move AWS account ID to environment variables**
   - Remove from source code
   - Load from `AWS_BEDROCK_MODEL_ARN` env var
   - Add validation at startup

3. **Add response validation**
   - Use `.get()` instead of direct key access
   - Validate response structure before parsing
   - Log full response on parse failure

4. **Fix ChromaDB path handling**
   - Use absolute path: `~/.tf_rag_chroma`
   - Create directory if missing
   - Add path to logging

5. **Add file write safety**
   - Create backups before overwriting
   - Check file permissions
   - Log all file operations

### Phase 2: High Priority (Before First Production Deployment)
6. **Implement proper logging**
   - Use Python logging module
   - Log to file: `tf_rag.log`
   - Include timestamps and severity levels

7. **Externalize configuration**
   - Create `config.yaml`
   - Support environment variable overrides
   - Document all settings

8. **Add startup validation**
   - Check AWS credentials
   - Verify Checkov availability
   - Validate ChromaDB accessibility

### Phase 3: Medium Priority (Next Iteration)
9. **Add input validation**
   - Limit prompt length (max 5000 chars)
   - Detect prompt injection patterns
   - Sanitize user input

10. **Improve error messages**
    - Include full context
    - Provide recovery suggestions
    - Link to documentation

---

## Files Requiring Changes

```
main.py
├── Add error handling to generate_terraform() ⚠️ CRITICAL
├── Move MODEL_ARN to environment variable ⚠️ CRITICAL
├── Add response validation ⚠️ CRITICAL
├── Implement logging ⚠️ HIGH
├── Add input validation ⚠️ HIGH
└── Fix file write safety ⚠️ HIGH

vector_store.py
├── Fix ChromaDB path handling ⚠️ CRITICAL
├── Add error handling ⚠️ HIGH
└── Add initialization validation ⚠️ HIGH

security_scanner.py
├── Improve error messages ⚠️ MEDIUM
└── Add timeout handling ⚠️ MEDIUM

New Files Required:
├── config.yaml - Configuration file
├── requirements.txt - Pinned dependencies (if not present)
└── .env.example - Environment variables template
```

---

## Testing Recommendations

1. **Error Scenarios**
   - Test with invalid AWS credentials
   - Test with unreachable Bedrock endpoint
   - Test with malformed model responses
   - Test with full disk/permission issues

2. **Edge Cases**
   - Empty user input
   - Very long prompts (5000+ chars)
   - Repeated violations (same error 5 times)
   - Checkov timeout scenario

3. **Integration Tests**
   - Full pipeline with mock Bedrock responses
   - ChromaDB persistence across runs
   - File operations and backups

---

## Security Concerns

| Concern | Severity | Status |
|---------|----------|--------|
| AWS account ID exposed | 🔴 HIGH | ❌ Not addressed |
| AWS credentials in code | 🔴 HIGH | ✓ Using IAM roles |
| User input not sanitized | 🟠 MEDIUM | ❌ Not addressed |
| API response not validated | 🟠 MEDIUM | ❌ Not addressed |
| Temp files from Checkov | 🟢 LOW | ✓ Cleaned up in finally |

---

## Next Steps

1. **Immediate** (Today):
   - Move AWS account ID to environment variables
   - Add basic error handling to Bedrock calls
   - Add response structure validation

2. **This Week**:
   - Implement logging system
   - Add configuration file
   - Add startup validation

3. **This Sprint**:
   - Comprehensive error handling
   - Unit tests for error scenarios
   - Security review and penetration testing

---

**Generated**: 2026-06-03  
**Reviewed by**: Code Analysis Agent  
**Status**: Pending Implementation
