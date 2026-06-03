# Interview Preparation Plan — Cloud AI Engineer (BCG via IWS)

**Candidate:** Digambar Rajaram  
**Role:** Cloud AI Engineer | Client: BCG | Employer: IWS  
**Date Prepared:** June 2026

---

> **Recruiter's Lens — Honest Assessment:**  
> Your DevOps/Infra background is solid. The gap the recruiter *will* probe is your AI/Bedrock experience — your resume says "experimentation" and "concepts", while the JD needs production-grade AWS Bedrock, RAG pipelines, and LLM integration. Expect 60–70% of interview questions to be on Bedrock, AI workflows, and RAG. Prepare honest, confident answers that bridge your hands-on DevOps strength with your AI learning journey. **Do not bluff — interviewers at BCG-level will deep-dive.**

---

## Section 1: Gap Analysis — JD vs Your Resume

| JD Requirement | Your Resume Status | Risk Level | Action |
|---|---|---|---|
| AWS Bedrock (production) | "Experimentation / concepts" | 🔴 HIGH | Build a working demo project |
| RAG / Vector Search pipelines | "Foundational architecture" | 🔴 HIGH | Implement with FAISS/ChromaDB |
| Terraform Cloud APIs | Terraform IaC (strong) | 🟢 LOW | Read TFC API docs |
| Python (AI workflows) | Python scripting (ops) | 🟡 MEDIUM | Write Bedrock Python scripts |
| LangChain / LangGraph | Listed in skills | 🟡 MEDIUM | Build a small agent |
| Kubernetes / EKS | Strong (production) | 🟢 LOW | Review EKS AI deployment |
| GitHub Actions / CI/CD | Strong (Jenkins focus) | 🟡 MEDIUM | Build a GitHub Actions pipeline |
| Vault / IRSA | IRSA done in prod | 🟢 LOW | Revisit Vault basics |
| Checkov / Sentinel / OPA | Not in resume | 🟡 MEDIUM | Read Checkov basics |

---

## Section 2: AWS Bedrock — Deep Dive (HIGHEST PRIORITY)

> The recruiter **will** go deep here. Prepare every concept below.

### 2.1 Core Concepts You Must Know Cold

- **What is Amazon Bedrock?**  
  Fully managed service to access foundation models (FMs) from AWS and third parties (Anthropic Claude, Meta Llama, Mistral, Titan, etc.) via a single API — no model hosting needed.

- **Bedrock vs SageMaker** — Know when to choose each:
  - Bedrock: Consume pre-built FMs, RAG, agents — no ML training infra
  - SageMaker: Custom model training, fine-tuning, full MLOps pipelines

- **Key Bedrock APIs:**
  - `InvokeModel` — single synchronous call
  - `InvokeModelWithResponseStream` — streaming responses
  - `Converse` / `ConverseStream` — multi-turn chat (new, model-agnostic)
  - `RetrieveAndGenerate` — built-in RAG with Knowledge Bases
  - Agents API — autonomous multi-step reasoning

### 2.2 Bedrock Knowledge Bases (RAG)

**Know this end-to-end — it's the core of the JD's RAG requirement.**

```
Flow: Document → Chunking → Embedding Model (Titan Embeddings) 
      → Vector Store (OpenSearch Serverless / Aurora pgvector / Pinecone) 
      → Bedrock Knowledge Base 
      → RetrieveAndGenerate API → LLM Response
```

- **Embedding models available:** Amazon Titan Embeddings, Cohere Embed
- **Vector stores supported:** OpenSearch Serverless, Aurora PostgreSQL (pgvector), Redis Enterprise, Pinecone, MongoDB Atlas
- **Chunking strategies:** Fixed-size, semantic, hierarchical
- **Sync process:** `StartIngestionJob` → monitors sync status

**Expected Interview Question:**  
*"How would you build a RAG pipeline for Terraform module retrieval using Bedrock?"*

**Your Answer Framework:**
1. Store Terraform module docs/READMEs in S3
2. Create Bedrock Knowledge Base, point to S3, choose OpenSearch Serverless as vector store
3. Bedrock auto-chunks, embeds with Titan, indexes to OpenSearch
4. At query time: user input → Bedrock `RetrieveAndGenerate` → retrieves top-k chunks → sends to Claude/Llama → returns answer
5. Wrap in Python Lambda or FastAPI service

### 2.3 Bedrock Agents

- **What they do:** Multi-step reasoning — decide which tool/action to call, execute, observe result, repeat (ReAct pattern)
- **Components:** Foundation model + Action Groups (Lambda functions) + Knowledge Bases + Memory
- **Action Group:** Define an OpenAPI schema → Bedrock calls your Lambda when it decides to use that action
- **Use case in this JD:** Agent receives natural language ("create an S3 bucket in us-east-1 with versioning") → generates Terraform → validates → submits to TFC

### 2.4 Bedrock Security (Critical for BCG)

- All data encrypted in transit (TLS) and at rest (KMS)
- **No data is used to train AWS models** — data isolation guaranteed
- **IAM permissions:** `bedrock:InvokeModel`, `bedrock:RetrieveAndGenerate`
- **VPC endpoints** for private connectivity
- **Guardrails:** Content filters (hate, violence, PII), topic denial, grounding checks — know how to configure these
- **Model access:** Must be explicitly enabled per region in the console

### 2.5 Python Code — You Must Be Able to Write This

```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Basic InvokeModel
def invoke_claude(prompt: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps(body)
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

# RAG with Knowledge Base
def rag_query(kb_id: str, question: str) -> str:
    bedrock_agent = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
    response = bedrock_agent.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
            }
        }
    )
    return response['output']['text']
```

---

## Section 3: RAG & Vector Search — Deep Dive

### 3.1 RAG Architecture (Retrieval-Augmented Generation)

```
[User Query] 
    → Embedding (convert query to vector)
    → Vector DB similarity search (cosine/dot product)
    → Top-K relevant chunks retrieved
    → Chunks + original query → LLM prompt
    → LLM generates grounded answer
```

**Why RAG?** LLMs have stale knowledge. RAG injects current, private, or domain-specific data without fine-tuning.

### 3.2 Key Components

| Component | Options |
|---|---|
| Embedding model | Amazon Titan, OpenAI Ada, Cohere Embed, HuggingFace |
| Vector database | OpenSearch, FAISS (local), pgvector, Pinecone, ChromaDB |
| Chunking | Fixed (512 tokens), sliding window, semantic |
| Retrieval | Dense (vector), Sparse (BM25), Hybrid |
| Re-ranking | Cohere Rerank, cross-encoders |

### 3.3 For This JD — Terraform Module RAG

**Use case:** User types "I need an EKS cluster with 3 node groups" → system retrieves relevant Terraform module docs → LLM generates Terraform config using real module signatures.

**Your talking points:**
- Index Terraform Registry docs + internal modules into vector store
- Query at runtime to find matching modules
- LLM generates config using actual module variable names (reduces hallucination)
- Validate generated HCL with `terraform validate` or Checkov before execution

### 3.4 Chunking Strategies — Know the Trade-offs

- **Fixed-size:** Simple, fast, may break semantic meaning mid-sentence
- **Semantic:** Better context, more expensive, uses sentence-transformers
- **Hierarchical:** Parent-child chunks — retrieve small, include large context

---

## Section 4: Terraform Cloud APIs — Integration Focus

### 4.1 TFC vs Open Source Terraform

| Feature | Terraform OSS | Terraform Cloud |
|---|---|---|
| State management | Manual (S3+DynamoDB) | Built-in |
| Remote runs | No | Yes |
| Policy (Sentinel/OPA) | No | Yes |
| Workspace management | No | Yes |
| VCS integration | No | Yes |

### 4.2 TFC API — Key Endpoints to Know

```
Base URL: https://app.terraform.io/api/v2/

# Workspaces
GET    /organizations/{org}/workspaces
POST   /organizations/{org}/workspaces
PATCH  /workspaces/{workspace_id}

# Runs (trigger a plan/apply)
POST   /runs
GET    /runs/{run_id}

# Plans & Applies
GET    /plans/{plan_id}
POST   /runs/{run_id}/actions/apply

# Variables
POST   /workspaces/{workspace_id}/vars
```

### 4.3 Python TFC API Example

```python
import requests

TFC_TOKEN = "your-token"
ORG = "your-org"
HEADERS = {
    "Authorization": f"Bearer {TFC_TOKEN}",
    "Content-Type": "application/vnd.api+json"
}

def create_workspace(name: str, tf_version: str = "1.6.0"):
    payload = {
        "data": {
            "type": "workspaces",
            "attributes": {
                "name": name,
                "terraform-version": tf_version,
                "auto-apply": False
            }
        }
    }
    resp = requests.post(
        f"https://app.terraform.io/api/v2/organizations/{ORG}/workspaces",
        headers=HEADERS,
        json=payload
    )
    return resp.json()

def trigger_run(workspace_id: str, message: str = "AI-triggered run"):
    payload = {
        "data": {
            "attributes": {"message": message, "auto-apply": False},
            "type": "runs",
            "relationships": {
                "workspace": {"data": {"type": "workspaces", "id": workspace_id}}
            }
        }
    }
    resp = requests.post(
        "https://app.terraform.io/api/v2/runs",
        headers=HEADERS,
        json=payload
    )
    return resp.json()
```

---

## Section 5: LangChain / LangGraph Basics

> Listed in your resume — be ready to explain at least basic usage.

### 5.1 LangChain Core Concepts

- **Chain:** Sequence of LLM calls or operations
- **LLM wrapper:** Abstracts Bedrock, OpenAI, etc. into same interface
- **Prompt Templates:** Reusable, parameterized prompts
- **Document loaders + Text splitters:** Ingest and chunk documents
- **Vector stores:** `FAISS`, `Chroma` integrations
- **Retrievers:** Interface to query vector stores
- **RetrievalQA Chain:** The standard RAG chain

### 5.2 LangGraph

- Graph-based agent framework built on LangChain
- Nodes = functions/LLM calls; Edges = conditional routing
- **Supports cycles** (unlike simple chains) — enables agentic loops (think, act, observe, repeat)
- Use when you need multi-step, stateful agents

### 5.3 Simple Bedrock + LangChain RAG

```python
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load and split docs
loader = TextLoader("terraform_modules.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
vectorstore = FAISS.from_documents(chunks, embeddings)

# RAG chain
llm = BedrockLLM(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

answer = qa_chain.invoke("How do I create an EKS cluster with IRSA?")
```

---

## Section 6: Policy & Validation (Checkov, Sentinel, OPA)

> You don't have production exp — be honest but show awareness.

### 6.1 Checkov

- **What:** Static analysis for Terraform (and other IaC)
- **How it works:** Scans `.tf` files or Terraform plan JSON for misconfigurations
- **Common checks:** S3 public access, unrestricted security groups, unencrypted EBS, missing MFA
- **Run:** `checkov -d ./terraform/` or `checkov -f main.tf`
- **In this JD:** Validate AI-generated Terraform before submitting to TFC

### 6.2 OPA / Rego

- **What:** General-purpose policy engine; policies written in Rego
- **In Terraform context:** Validate plan output against business rules
- **Example policy:** "No EC2 instance larger than t3.large in dev"

### 6.3 Sentinel (TFC native)

- HashiCorp's policy-as-code framework, built into TFC
- Runs automatically between `plan` and `apply`
- Policies can `advisory` (warn), `soft-mandatory` (override with reason), or `hard-mandatory` (block)

**Your Answer:**  
*"I haven't used Checkov in production but I understand its role — in this platform, AI-generated Terraform configs would be validated through Checkov for security misconfigs and Sentinel policies in TFC before any apply is triggered. This is a guardrail layer I'm actively learning."*

---

## Section 7: Kubernetes / EKS — Deploying AI Workloads

You have strong EKS experience. The AI-specific questions:

- **How would you deploy a Bedrock-based API service on EKS?**
  - FastAPI/Flask app calling Bedrock → Dockerize → ECR → Helm chart → EKS
  - IRSA for the pod to assume IAM role with `bedrock:InvokeModel` permission
  - HPA for scaling based on request load
  - Secrets Manager or Vault for any API keys

- **Resource management for AI workloads:**
  - LLM inference is CPU/memory heavy — define proper `requests` and `limits`
  - Consider GPU node pools if running local models (Ollama)

- **Service mesh considerations:**
  - mTLS between services if handling sensitive prompts/outputs

---

## Section 8: GitHub Actions CI/CD for AI Platform

### Sample Pipeline — AI-Driven Terraform Provisioning

```yaml
name: AI Terraform Provisioning

on:
  workflow_dispatch:
    inputs:
      user_request:
        description: 'Natural language infra request'
        required: true

jobs:
  generate-terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: us-east-1

      - name: Generate Terraform via Bedrock
        run: python scripts/generate_tf.py "${{ github.event.inputs.user_request }}"

      - name: Validate with Checkov
        run: checkov -d ./generated_tf/ --framework terraform

      - name: Submit to Terraform Cloud
        run: python scripts/tfc_submit.py
        env:
          TFC_TOKEN: ${{ secrets.TFC_TOKEN }}
```

---

## Section 9: Vault & IRSA

### IRSA (you have production experience)

Be ready to explain clearly:

1. Create OIDC provider in IAM pointing to EKS cluster OIDC URL
2. Create IAM role with trust policy scoped to specific namespace + service account
3. Annotate Kubernetes service account with `eks.amazonaws.com/role-arn`
4. Pod inherits IAM role via projected token — no static credentials

**For Bedrock:** Pod's IAM role needs `bedrock:InvokeModel` and `bedrock:RetrieveAndGenerate` permissions.

### Vault Basics (study needed)

- HashiCorp Vault: secrets management, dynamic credentials
- **AWS Secrets Engine:** Vault generates temporary AWS IAM credentials on demand
- **Kubernetes Auth:** Pods authenticate to Vault using their K8s service account token
- **vs IRSA:** IRSA is AWS-native and simpler for AWS-only setups; Vault is cloud-agnostic and adds secret rotation

---

## Section 10: Anticipated Interview Questions & Strong Answers

### Tier 1 — Very High Probability (AWS Bedrock Focus)

**Q1: "Walk me through how you've used AWS Bedrock."**

> "In my personal projects, I've built an AI-assisted incident management workflow where I used Bedrock's Claude model via `InvokeModel` API to analyze CloudWatch logs and suggest root cause hypotheses. I also designed a RAG architecture where operational runbooks were chunked, embedded using Titan Embeddings, indexed into a vector store, and retrieved at query time to provide contextually relevant responses. I'll be honest — this was experimental/personal project work, not production at Protean. At Protean, my AI exposure was using ChatGPT and Claude for productivity. I'm actively building production-ready patterns and can speak to the architecture deeply."

**Q2: "Explain how you'd build a RAG pipeline for Terraform module retrieval."**

> Use the framework from Section 3.3 above. Draw the full flow. Mention chunking strategy choice (semantic for docs), embedding model (Titan), vector store (OpenSearch Serverless via Bedrock KB), retrieval (top-3 chunks), and how it feeds into the LLM prompt. Mention hallucination reduction by grounding output in real module signatures.

**Q3: "What is the difference between Bedrock Agents and Bedrock Knowledge Bases?"**

> "Knowledge Bases are for retrieval — you index documents and retrieve relevant context for RAG. Agents are for action — they use an LLM to reason over multiple steps, decide which tools/APIs to call (via Action Groups mapped to Lambda functions), and loop until the task is complete. For this platform, I'd use both: KB for retrieving relevant Terraform modules, and Agents for the multi-step workflow of generate → validate → submit."

**Q4: "How would you prevent an AI-generated Terraform config from doing something dangerous?"**

> Multi-layer guardrails:
> 1. **Prompt-level:** System prompt constrains the LLM to only use approved modules/providers
> 2. **Bedrock Guardrails:** Topic denial, PII filtering on inputs/outputs
> 3. **Checkov scan:** Static analysis catches misconfigs (public S3, open SGs)
> 4. **Sentinel/OPA policy:** TFC policy-as-code blocks non-compliant resources
> 5. **Plan review:** Always generate a plan first; apply requires explicit approval
> 6. **IRSA least-privilege:** TFC role only has permissions for allowed resource types

**Q5: "What embedding model would you use and why?"**

> "For Terraform docs — Amazon Titan Embeddings v2 is the natural choice within the Bedrock ecosystem since it integrates natively with Bedrock Knowledge Bases and OpenSearch Serverless. For a custom pipeline, I'd benchmark Titan against Cohere Embed v3 on domain-specific retrieval quality. Terraform docs have structured, technical language so embedding quality on code/technical text matters."

### Tier 2 — High Probability (Platform Engineering)

**Q6: "How do you trigger a Terraform Cloud run programmatically?"**

> Walk through the TFC API: authenticate with team token → POST to `/api/v2/runs` with workspace ID → poll run status → handle plan output → trigger apply if validated. See Section 4.2/4.3 code.

**Q7: "You have Jenkins experience. Why would you choose GitHub Actions for this platform?"**

> "GitHub Actions co-locates CI/CD config with the code repository, has native GitHub integration (PR triggers, PR comments with plan output), better marketplace ecosystem for Terraform/AWS actions, and no infra to maintain. Jenkins makes sense for complex enterprise setups with many existing integrations — but for a new AI platform where the workflow is tightly tied to git events, GitHub Actions is cleaner."

**Q8: "How would you handle secrets in this pipeline?"**

> - TFC token: GitHub Actions secret → never in code
> - AWS credentials: OIDC federation between GitHub Actions and AWS IAM (no static keys)
> - Application secrets: IRSA for pods to access Secrets Manager or Vault
> - Bedrock: No API keys needed — IAM role-based access

### Tier 3 — Medium Probability (Architecture / Scenario)

**Q9: "Design the end-to-end architecture for this AI provisioning platform."**

```
User (Slack/Web UI)
    → API Gateway → Lambda/FastAPI
    → Bedrock Agent
        → Action: RAG lookup (Bedrock KB → OpenSearch → Terraform modules)
        → Action: Generate Terraform HCL (Claude via Bedrock)
        → Action: Validate (Checkov Lambda)
        → Action: Submit to TFC (TFC API Lambda)
    → TFC: Sentinel policy check → Plan → Approval gate → Apply
    → Notifications (Slack)
    → Audit log (CloudTrail + S3)
```

**Q10: "How would you handle LLM hallucinations in Terraform generation?"**

> 1. Retrieval grounding (RAG) — LLM only uses real module signatures
> 2. Few-shot examples in system prompt — show correct HCL format
> 3. Temperature = 0 for deterministic output
> 4. Post-generation validation — `terraform validate`, Checkov
> 5. Human approval gate before apply (especially for prod)
> 6. Structured output prompting — ask LLM to return JSON with HCL + reasoning

---

## Section 11: "Honest Bridge" Phrases to Use in Interview

When you lack direct production experience in AI, use these frames:

- *"I've implemented this in personal/experimental projects and can speak to the architecture. I'm confident in getting this to production quality quickly given my background with [related skill]."*
- *"At Protean, my AI exposure was primarily as a user of LLM tools for productivity. My hands-on AI engineering has been through self-driven projects, which I'm happy to walk through in detail."*
- *"I understand the concept and have worked with it in a lab setup. The production hardening aspects — monitoring, fallback handling, cost controls — are things I'd want to learn from the team's existing patterns."*

**Never say:** "I know this" when you don't. BCG-level interviewers will probe immediately and a bluff destroys credibility.

---

## Section 12: Study & Build Plan (Pre-Interview)

### Week 1 — Bedrock Hands-On

- [ ] Enable Claude 3 Sonnet in AWS Bedrock console (us-east-1)
- [ ] Write Python script calling `InvokeModel` and `ConverseStream`
- [ ] Create a Bedrock Knowledge Base with a sample S3 document set
- [ ] Test `RetrieveAndGenerate` API from Python
- [ ] Configure a Bedrock Guardrail (content filter + topic denial)

### Week 2 — RAG Pipeline

- [ ] Build local RAG with LangChain + FAISS (no cloud cost)
- [ ] Replace with Bedrock Embeddings + Bedrock KB
- [ ] Load 5–10 Terraform module READMEs as your knowledge base
- [ ] Test: query "create EKS cluster" → retrieve relevant module docs → generate config

### Week 3 — Platform Integration

- [ ] Set up a free Terraform Cloud account, create a workspace
- [ ] Write Python to trigger a TFC run via API
- [ ] Build a GitHub Actions pipeline: input → Bedrock call → TFC trigger
- [ ] Run Checkov on a sample Terraform file, fix findings

### Week 4 — Polish & Mock

- [ ] Record yourself answering Q1–Q5 from Section 10
- [ ] Review your resume — be ready to explain every line with specifics
- [ ] Prepare 3 STAR-format stories from Protean experience
- [ ] Study BCG's technology practice areas (Digital, Tech Build)

---

## Section 13: Questions to Ask the Interviewer

- *"What does the current state of the platform look like — are you building from scratch or enhancing an existing system?"*
- *"What LLM models are you using via Bedrock today — Claude, Titan, or others?"*
- *"How is the team handling LLM output validation and cost control at scale?"*
- *"What does a typical sprint look like — are engineers owning full features end-to-end?"*
- *"Is the expectation to be embedded with BCG teams directly, or working through IWS delivery?"*

---

## Quick Reference Cheat Sheet

| Topic | Key Point |
|---|---|
| Bedrock vs SageMaker | Bedrock = consume FMs; SageMaker = train/fine-tune |
| RAG flow | Doc → Chunk → Embed → Vector DB → Retrieve → LLM |
| Bedrock KB vector stores | OpenSearch Serverless, Aurora pgvector, Pinecone |
| TFC API auth | Bearer token in `Authorization` header |
| Checkov | `checkov -d ./` — IaC static analysis |
| IRSA | OIDC provider → IAM role → K8s service account annotation |
| Guardrails | Content filter + topic denial + grounding + PII |
| LangGraph vs LangChain | LangGraph = stateful agents with cycles; LangChain = chains |
| Temperature for Terraform | Set to 0 — need deterministic output |
| Sentinel enforcement | advisory / soft-mandatory / hard-mandatory |

---

*Prepared for: Digambar Rajaram | Role: Cloud AI Engineer | Client: BCG*
