# Cloud AI Engineer — Interview Battle Plan
**Role:** Cloud AI Engineer | **Interview:** Wed Jun 4, 2026 · 16:30  
**Verdict:** You have 80% of this JD in production. Close the Bedrock + RAG gap in 2 days.

---

## Your Skills vs the JD

| Skill | Status | Your Evidence |
|---|---|---|
| Terraform + EKS | ✅ Production | 2 EKS clusters, 20–40 nodes, Protean |
| GitHub Actions + CI/CD | ✅ Production | AI-Driven DevOps project + Jenkins pipelines |
| IRSA / OIDC | ✅ Production | Pod-level IAM on EKS at Protean |
| Docker + Helm + Kubernetes | ✅ Production | eSign/eKYC microservices |
| Python scripting | ✅ Production | Automation scripts, ~40% effort reduction |
| AWS Bedrock | ⚡ Build demo | 2-day demo this week |
| RAG + Vector Search | ⚡ Build demo | ChromaDB locally, zero cost |
| Checkov / OPA | ⚡ Learn | Checkov CLI in demo, understand OPA concept |
| Terraform Cloud APIs | ⚡ Understand | Read TFC REST API docs, no build needed |

---

## Day-by-Day Plan

### Monday Jun 2 — Build the Core Demo (3–4 hrs)

**Goal:** Python script that calls Bedrock and generates Terraform code from a text prompt.

**Steps:**
1. Enable AWS Bedrock in your account → request access to Claude Sonnet (takes ~5 min)
2. Write a 30-line Python script: user types `"create an S3 bucket"` → Bedrock returns `.tf` code
3. Add Checkov scan on the output — 5 lines of Python to call checkov CLI and parse results
4. Save output as `generated.tf` — you now have end-to-end proof: **prompt → code → scanned**

**Starter code:**
```python
import boto3
import json
import subprocess

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def generate_terraform(user_prompt: str) -> str:
    system_prompt = """You are a Terraform expert. When given an infrastructure request,
    output ONLY valid Terraform HCL code. No explanations, no markdown fences."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    })

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=body
    )
    return json.loads(response["body"].read())["content"][0]["text"]

def scan_with_checkov(tf_code: str) -> dict:
    with open("generated.tf", "w") as f:
        f.write(tf_code)
    result = subprocess.run(
        ["checkov", "-f", "generated.tf", "--output", "json"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout) if result.stdout else {"error": result.stderr}

if __name__ == "__main__":
    prompt = input("Describe your infrastructure: ")
    print("\n⚙️  Generating Terraform...\n")
    tf_code = generate_terraform(prompt)
    print(tf_code)
    print("\n🔍 Running Checkov security scan...\n")
    scan_result = scan_with_checkov(tf_code)
    passed = scan_result.get("summary", {}).get("passed", 0)
    failed = scan_result.get("summary", {}).get("failed", 0)
    print(f"✅ Passed: {passed}  ❌ Failed: {failed}")
```

**Install dependencies:**
```bash
pip install boto3 checkov
aws configure  # use your personal account credentials
```

---

### Monday Evening — Add RAG (2 hrs)

**Goal:** Demo retrieves your own Terraform module docs before calling the LLM.

**Why this matters:** This is the exact architecture the JD describes. Even a local version proves you understand it.

**Steps:**
1. Use ChromaDB (local, zero cost) — store 3–4 sample Terraform module docs as vectors
2. Before calling Bedrock, retrieve the top-1 relevant doc
3. Inject it into the system prompt so the LLM uses your org's exact module spec

**RAG code:**
```python
import chromadb
from chromadb.utils import embedding_functions

# Sample internal module docs (in real life, these come from S3/Confluence)
TERRAFORM_DOCS = [
    {
        "id": "eks-module",
        "text": """EKS Module: Use module source 'terraform-aws-modules/eks/aws'.
        Required vars: cluster_name, cluster_version='1.29', vpc_id, subnet_ids.
        Always enable IRSA: enable_irsa=true. Node groups min=2, max=10.
        Tags: Environment, Team, CostCenter are mandatory."""
    },
    {
        "id": "s3-module",
        "text": """S3 Module: Use aws_s3_bucket resource. Always set:
        server_side_encryption_configuration with AES256.
        Block all public access. Enable versioning for production buckets.
        Never use ACLs — use bucket policies only."""
    },
    {
        "id": "vpc-module",
        "text": """VPC Module: Use module source 'terraform-aws-modules/vpc/aws'.
        Always create 3 public + 3 private subnets across AZs.
        Enable NAT gateway (single_nat_gateway=false for prod).
        Enable VPC flow logs."""
    },
]

def setup_vector_store():
    client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection("tf_modules", embedding_function=ef)
    collection.add(
        documents=[d["text"] for d in TERRAFORM_DOCS],
        ids=[d["id"] for d in TERRAFORM_DOCS]
    )
    return collection

def retrieve_context(collection, query: str) -> str:
    results = collection.query(query_texts=[query], n_results=1)
    return results["documents"][0][0] if results["documents"] else ""

# In your generate_terraform function, add context injection:
def generate_terraform_with_rag(user_prompt: str, collection) -> str:
    context = retrieve_context(collection, user_prompt)
    system_prompt = f"""You are a Terraform expert for this organization.
    Use the following internal module specification:

    {context}

    Output ONLY valid Terraform HCL using these exact module specs. No explanations."""

    # ... rest of Bedrock call same as before
```

**Install:**
```bash
pip install chromadb
```

---

### Tuesday Jun 3 — Interview Narrative (2 hrs)

**Goal:** For every JD skill, have a 2-sentence answer using YOUR real work.

#### Prepared answers by topic

**"Tell me about your Terraform experience"**
> "At Protean I authored Terraform IaC to provision and manage EKS clusters — 2 clusters with 20–40 EC2 worker nodes each — along with full VPC, ALB, IAM, and S3+DynamoDB remote state with locking. I've handled multi-AZ architectures, IRSA/OIDC for pod-level IAM, and KMS encryption in production supporting 60M+ users."

**"How would you handle the RAG pipeline for Terraform modules?"**
> "I'd store module schemas and runbooks as vector embeddings — either in Amazon OpenSearch Serverless or locally with ChromaDB for a prototype. When a user submits a prompt, I retrieve the top-k relevant module docs and inject them into the Bedrock system prompt, so the LLM generates code that matches our org's exact specs rather than generic Terraform."

**"How do you secure the AI-generated Terraform before it hits production?"**
> "Two layers: Checkov for static scanning of the generated .tf file — catches unencrypted S3, open security groups, missing tags. And OPA/Sentinel for org-wide policy-as-code in Terraform Cloud, which runs server-side before any plan executes. Bedrock Guardrails also filters the input prompt to block requests for overly expensive or unauthorized resources."

**"Walk me through your CI/CD experience with GitHub Actions"**
> "My AI-Driven DevOps project on GitHub uses OIDC auth with GitHub Actions — no stored AWS credentials. The pipeline evaluates CloudWatch logs, correlates CI/CD deployment history, and triggers incident assessment. At Protean I built Jenkins pipelines for EKS: GitHub webhook → Maven build → Docker image → ECR push → Helm deploy with rollback triggers."

**"What is IRSA and why does it matter here?"**
> "IRSA — IAM Roles for Service Accounts — lets Kubernetes pods assume AWS IAM roles via OIDC without any hardcoded credentials. I implemented this at Protean for our EKS clusters. In this platform it's critical because the Python AI application pod needs to call Bedrock and S3 securely — IRSA gives it scoped, temporary credentials automatically."

---

### Tuesday Evening — Understand TFC + OPA (1 hr)

**Goal:** Conceptual clarity only — no build needed.

**Terraform Cloud APIs — what you need to know:**
- TFC exposes REST APIs: you call `/runs` to trigger a plan/apply
- Same pattern as triggering a Jenkins job via API — your Python app just POSTs to TFC
- Key endpoints: create workspace, upload config, trigger run, poll status
- In the platform: after Checkov passes, Python calls TFC API to apply — no human needed

**Checkov vs OPA — how to explain the difference:**
> "Checkov is the fast static scan that runs on the raw `.tf` file before anything reaches Terraform Cloud. OPA/Sentinel is the policy-as-code layer inside Terraform Cloud itself — it enforces org-wide rules server-side before any plan can execute. Think of Checkov as the developer-side gate and Sentinel as the platform-side gate."

---

## Wednesday — Interview Day

### Your opening 30-second pitch

> "I bring 2+ years of production DevOps at Protean eGov, where I managed infrastructure supporting 60 million NPS subscribers — running EKS clusters, Terraform IaC, IRSA, and GitHub Actions in production. The GenAI layer was the only gap, so this week I built a working Bedrock + RAG demo that generates and Checkov-scans Terraform code from a plain text prompt. I'm ready to take that from prototype to production-grade platform."

### What to show

1. **Your GitHub repo** — the Bedrock + RAG + Checkov demo (even 80 lines of working code)
2. **Live run** — type "create an EKS cluster" → show the generated `.tf` → show the Checkov output
3. **Architecture diagram** — the 4-layer flow: Prompt → RAG → Bedrock → Checkov → TFC

### How to handle "you're new to Bedrock"

> "You're right that Bedrock is new to me — I built my first integration this week. But the pattern is identical to what I do with Jenkins APIs and Terraform Cloud: call an endpoint, parse the response, handle errors, integrate into a pipeline. The infrastructure surrounding it — EKS, IRSA, GitHub Actions, Terraform — I've run all of that in production."

---

## Architecture You're Building (Reference)

```
[Developer Prompt]
       │
       ▼
[Python Agent] ──► [ChromaDB / OpenSearch] (RAG retrieval)
       │                    │
       │◄───────────────────┘ (inject module context)
       ▼
[AWS Bedrock - Claude] (generates .tf code)
       │
       ▼
[Checkov / OPA scan] ──► (fail: loop back to LLM to fix)
       │ (pass)
       ▼
[Terraform Cloud API] ──► [AWS Infrastructure]
       │
       ▼
[EKS cluster] (hosts the whole Python application)
```

**Team structure you're joining:**

```
App/Product Teams          (consumers — use the platform)
        │
        ▼
Platform Engineering       (YOU — build & maintain the AI engine)
  - Cloud AI Engineers (Python, Bedrock, RAG, EKS)
  - DevOps Engineers   (GitHub Actions, Terraform Cloud)
        │
        ▼
Cloud Security / SecOps    (define OPA rules, manage Vault, Guardrails)
```

---

## Quick Reference — Cost Estimate for Your Demo

| Service | Usage | Approx Cost |
|---|---|---|
| AWS Bedrock (Claude Sonnet) | ~50 test calls | < $0.50 |
| ChromaDB | Local / free | $0 |
| Checkov | Open source CLI | $0 |
| EKS | Not needed for demo | $0 |

**Total demo cost: under $1.**

---

*Good luck on Wednesday. You have the infrastructure foundation — show them the AI layer is already in progress.*
