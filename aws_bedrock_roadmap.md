# AWS Bedrock Learning Roadmap

A structured, phase-by-phase guide to mastering Amazon Bedrock — from foundations to production-grade AI infrastructure.

---

## Overview

| Phase | Focus Area | Key Outcome |
|-------|-----------|-------------|
| [Phase 1](#phase-1--foundations) | Foundations | Understand LLMs, prompting, and model landscape |
| [Phase 2](#phase-2--core-bedrock) | Core Bedrock | Build RAG chatbots with Knowledge Bases and Agents |
| [Phase 3](#phase-3--devops-integration) | DevOps Integration | Wrap everything in Terraform and CI/CD pipelines |
| [Phase 4](#phase-4--production--security) | Production & Security | Harden with VPC, IAM, cost control, and reliability |
| [Phase 5](#phase-5--expert-specialisation) | Expert Specialisation | Multi-agent patterns, fine-tuning, MLOps, and certifications |

---

## Phase 1 — Foundations

> Understand what you're working with before touching Bedrock. These are the mental models that make everything else make sense.

### How LLMs Work

- **Tokens & tokenisation** — not words, tokens
- **Context window** — what it is and why it limits you
- **Transformer architecture intuition** — attention = focus
- **Training vs inference** — you only do inference
- **Parameters and weights** — why bigger ≠ always better
- **Temperature, top-p, top-k** — what they actually do to output

### Prompt Engineering

- Zero-shot vs few-shot prompting
- Chain-of-thought (CoT) — tell it to think step by step
- Role prompting — system vs user messages
- Instruction clarity — specificity beats length
- Output formatting — JSON, XML, bullet control
- Negative prompting — telling it what NOT to do
- Common failure modes: hallucination, repetition, drift

### Foundation Model Landscape

| Model | Provider | Notes |
|-------|----------|-------|
| Claude (Opus, Sonnet, Haiku) | Anthropic | Tiered by capability/cost |
| Titan | Amazon | Native AWS models |
| Llama 3 | Meta | Open weights |
| Mistral | Mistral AI | Efficient European models |
| Stable Diffusion / Titan Image | Various | Image generation |

**Decision rule:** Choose based on cost vs capability tradeoff for your specific use case.

### Phase 1 Project

> Write 10 prompts for a DevOps use case (incident summaries, runbook generation). Test the same prompt across Claude Haiku and Sonnet. Document quality vs cost difference.

---

## Phase 2 — Core Bedrock

> This is the heart of Bedrock. Every feature maps directly to something you'll use in production.

### Bedrock Architecture

- How Bedrock sits in your AWS account (no model hosting)
- Bedrock vs SageMaker — serverless inference vs managed clusters
- Model access — enabling models per region
- `InvokeModel` API — the core call via boto3
- `InvokeModelWithResponseStream` — streaming responses
- **Converse API** — unified multi-model interface *(prefer this)*
- Cross-region inference — latency and failover

### Knowledge Bases (RAG)

**Core concepts:**

- What RAG is — Retrieval Augmented Generation
- Vector embeddings — how text becomes searchable numbers
- Chunking strategies — fixed, semantic, hierarchical

**Implementation components:**

- S3 as your data source (PDF, txt, docx, HTML)
- Titan Embeddings model — converts docs to vectors
- OpenSearch Serverless as vector store (managed by Bedrock)
- Sync job — how Bedrock indexes your S3 bucket

**APIs:**

- `RetrieveAndGenerate` — full RAG in one call
- `Retrieve` — just get relevant chunks (custom pipeline)
- Metadata filtering — restrict retrieval by tags

### Bedrock Agents

> Agent = LLM + tools + memory + orchestration

- **Action Groups** — Lambda functions the agent can call
- **OpenAPI schema** — how you describe your Lambda to the agent
- **Agent aliases & versions** — like Lambda aliases
- **Session management** — multi-turn conversation state
- **Advanced prompts** — override default agent instructions
- **Trace & debug** — seeing step-by-step agent reasoning
- **Return of control** — agent hands back to your app mid-flight

### Guardrails

| Guardrail Type | Description |
|----------------|-------------|
| Content filters | Hate, violence, sexual content levels |
| Denied topics | Custom topics to block |
| Word filters | Block specific terms |
| Sensitive info redaction | PII masking (email, SSN, etc.) |
| Grounding check | Detect hallucinations vs knowledge base |

Guardrails can be applied to both models **and** agents.

### Phase 2 Project

> Build a RAG chatbot with an S3 bucket containing 5 internal docs. Use Bedrock Knowledge Base + Titan Embeddings. Call the `RetrieveAndGenerate` API from a Python script. Add a Guardrail that blocks PII.

---

## Phase 3 — DevOps Integration

> Your existing Jenkins, Terraform, and EKS knowledge gives you a head start most ML engineers lack.

### Terraform for Bedrock

Key resources to provision:

- `aws_bedrock_knowledge_base`
- `aws_bedrock_agent` + `aws_bedrock_agent_alias`
- `aws_opensearchserverless_collection` (vector store)
- `aws_iam_role` with Bedrock trust policy
- S3 bucket + bucket policy for knowledge base
- State management — same S3 + DynamoDB pattern
- Module structure for reusable Bedrock infra

### CI/CD for AI Workloads

- Jenkins pipeline: test prompt → deploy Lambda action group
- GitHub Actions OIDC for Bedrock API calls
- **Prompt versioning** — store prompts in Git like code
- Model evaluation in pipeline — automated quality checks
- Blue/green agent deployment via aliases
- Rollback strategy — point alias to previous version
- Testing AI outputs — non-deterministic test patterns

### Lambda Integration

- Lambda as Bedrock Agent action group
- Bedrock → Lambda → external API pattern
- Streaming responses through Lambda → API Gateway
- Lambda Powertools for tracing Bedrock calls
- Cold start impact on agent response latency
- Lambda layers for boto3 version pinning

### Observability for AI

| Tool | What to Monitor |
|------|----------------|
| CloudWatch Metrics | `InputTokenCount`, Latency |
| CloudWatch Logs Insights | Query Bedrock call logs |
| X-Ray Tracing | Agent → Lambda → API chain |
| Grafana Dashboard | Token usage + cost + latency |
| ElastAlert2 | Hallucination spike detection |
| Cost Anomaly Detection | Bedrock token budgets |

### Phase 3 Project

> Take your Phase 2 RAG chatbot and wrap it in IaC. Full Terraform module. Jenkins pipeline that syncs S3 docs, triggers Knowledge Base sync job, runs a test prompt, and alerts Slack if quality degrades.

---

## Phase 4 — Production & Security

> Most Bedrock tutorials skip this. Your infrastructure background means you already understand why this matters.

### IAM for Bedrock

- Bedrock service role — least privilege pattern
- Resource-based policies for cross-account access
- **IRSA** — pod-level Bedrock access from EKS
- Conditions: limit to specific model IDs only
- Deny policies — prevent unauthorised model access
- Session tags for multi-tenant access control

### Networking & VPC

- VPC endpoint for Bedrock (`bedrock-runtime`)
- VPC endpoint for S3 (knowledge base data source)
- Private subnet agent — no internet egress
- Security groups for Lambda inside VPC
- NACLs for Bedrock endpoint subnets
- **PrivateLink** — Bedrock traffic never leaves AWS backbone

### Cost Control

| Lever | Description |
|-------|-------------|
| Token pricing | Input vs output cost per model |
| On-demand vs provisioned throughput | Choose based on usage patterns |
| Bedrock model units (MUs) | When to buy dedicated capacity |
| Caching | Reduce repeat token costs |
| Cost allocation tags | Tag Bedrock API calls for attribution |
| AWS Budgets | Alert on token spend thresholds |
| Model selection | Haiku vs Sonnet as cost lever |

### Reliability Patterns

- Throttling — handling 429s with exponential backoff
- Multi-region failover for Bedrock endpoints
- Circuit breaker in Lambda for Bedrock unavailability
- Knowledge base sync failure handling
- Agent timeout patterns — max 60s per action group call
- Dead letter queue for async Bedrock invocations

### Phase 4 Project

> Harden your Phase 3 pipeline. Add VPC endpoint, IRSA for EKS pods, cost budget alert, CloudWatch alarm for latency P99 > 5s. Write a runbook for Bedrock throttling incidents.

---

## Phase 5 — Expert Specialisation

> This is where you differentiate. Most candidates stop at Phase 2. These patterns show you can design and operate production AI systems.

### Advanced Agent Patterns

- **Multi-agent orchestration** — supervisor + sub-agents
- **Inline agents** — dynamic action groups at runtime
- **Agent with code interpreter** — let it write and run code
- **Memory** — conversation history across sessions
- **Flows** — visual no-code orchestration of Bedrock components
- **Prompt chaining** — multi-step reasoning pipelines

### Model Evaluation & Fine-Tuning

- Bedrock Model Evaluation — automated benchmark jobs
- Human evaluation workflow integration
- Custom metrics for your domain (not just BLEU/ROUGE)
- Fine-tuning on Bedrock — when it's worth it
- Continued pre-training vs instruction fine-tuning
- Training data format — JSONL structure
- Model customisation job — Terraform provisioning

### MLOps Patterns

| Pattern | Description |
|---------|-------------|
| Prompt management | Bedrock Prompt Management service |
| Model versioning | Alias pinning in prod |
| A/B testing | Weighted aliases across model versions |
| Shadow mode | Run new model in parallel silently |
| Drift detection | Monitor output quality over time |
| Feedback loop | Human rating → fine-tuning dataset |

### Certifications & Portfolio

**Certifications (in order):**

1. **AWS Certified AI Practitioner (AIF-C01)** — get this first
2. **AWS ML Specialty** — deeper theory, worth having

**Target Roles:**
- DevOps AI/ML Engineer
- Platform AI Engineer

**Portfolio checklist:**
- 3 projects on GitHub with Terraform + CI/CD
- Blog post: *"How I secured Bedrock in a production VPC"*
- LinkedIn headline: DevOps + AI Infrastructure

### Phase 5 Capstone Project

> **AI-Powered Incident Management Tool**
>
> An EKS microservice calls a Bedrock Agent with 2 action groups:
> - Query CloudWatch via Lambda
> - Create Jira ticket via Lambda
>
> **Requirements:** Full Terraform · Jenkins CI/CD · VPC endpoint · Grafana dashboard · Guardrails enabled
>
> This is your portfolio centrepiece.

---

## Quick Reference — Key API Calls

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Converse API (preferred — works across all models)
response = client.converse(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{"role": "user", "content": [{"text": "Your prompt here"}]}]
)

# RetrieveAndGenerate (RAG in one call)
bedrock_agent = boto3.client("bedrock-agent-runtime")
response = bedrock_agent.retrieve_and_generate(
    input={"text": "Your question"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "YOUR_KB_ID",
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet..."
        }
    }
)
```

---

*Roadmap covers AWS Bedrock from first principles to production. Each phase builds on the last — complete the phase project before moving forward.*
