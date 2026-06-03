# 30-Minute Interview — Question Bank & Strategy
## Cloud AI Engineer | BCG via IWS | Digambar Rajaram

---

## Who Will Be on the Call — People Intelligence

Based on research, all CC'd email IDs on the interview invite are **IWS HR/Resource team**, not BCG technical interviewers. Here's what was found:

| Person | Role | What They'll Do |
|---|---|---|
| **Tejaswi CK** (interviewer) | HR Executive, IWS | Intro, HR questions, screening |
| **Sabrina Sookias** | Senior Executive – Resource Management, IWS | Likely observing/evaluating fit for placement |
| **Meenakshi Chug** | HR Recruiter, IWS | Talent acquisition, may probe tech at surface level |
| **Sumit Kumar** | Senior HR Executive, IWS | Panel observer or second HR screener |
| **Aditi / Yashi Srivastava** | IWS HR team members | Panel support |

> **Key Insight:** The subject line says **"Internal Interview"** — this is IWS's own screening round **before** they submit you to BCG. This is not yet a BCG technical round. The panel is mostly HR, with possibly one technical person to gauge baseline.
>
> **What this means for you:** This round will assess your clarity, confidence, communication, and whether your resume matches your answers. They are deciding if you're safe to submit to BCG's technical panel. Expect a **mix of HR + surface-level technical** questions, not deep architecture dives. Those come in the BCG round.

---

## 30-Minute Interview Timeline

```
00:00 – 03:00  →  Introductions, your 2-minute self-intro
03:00 – 10:00  →  HR & Motivation questions (3–4 questions)
10:00 – 22:00  →  Technical screening (5–7 questions)
22:00 – 28:00  →  Scenario/situational questions (1–2 questions)
28:00 – 30:00  →  Your questions to them
```

---

## PART 1 — HR & Motivation Questions

### Q1. "Tell me about yourself." ⭐ ALWAYS ASKED
**Difficulty:** Easy  
**Time to answer:** 90–120 seconds max

**Your scripted answer:**
> "I'm Digambar, a DevOps and Cloud Engineer with 2.4 years of production experience at Protean eGov, where I managed large-scale infrastructure across AWS and VMware — covering EKS, Terraform, CI/CD pipelines, and observability. I supported platforms that serve 60 million NPS subscribers and 300 million PAN cardholders, so reliability and automation were critical. Over the last several months, I've been actively upskilling into AI-driven cloud engineering — working with AWS Bedrock APIs, building RAG pipelines, and experimenting with LangChain and LLM workflows. This Cloud AI Engineer role at BCG is the exact intersection I'm targeting — where my DevOps infrastructure depth meets the AI integration work I've been building toward."

---

### Q2. "Why are you leaving Protean? You just finished 2.4 years." ⭐ LIKELY ASKED
**Difficulty:** Easy–Medium  
**Watch out:** Don't badmouth. Don't sound unstable.

**Your answer:**
> "Protean was a great foundation — large-scale infra, real production pressure. But the work was primarily traditional DevOps. I want to move toward AI-driven infrastructure, which isn't the direction Protean is heading. This role is exactly the transition I've been preparing for — building AI-powered provisioning workflows, Bedrock integrations, RAG pipelines. It's a deliberate career move, not just a jump."

---

### Q3. "You mentioned 'hands-on exposure' to Bedrock — but have you actually worked with it in production?"
**Difficulty:** Medium — this is a trap question
**Be honest.**

> "At Protean, AI tooling wasn't part of the scope — my hands-on Bedrock experience is from personal projects and self-study over the last few months. I've invoked Claude via the Bedrock API, built a RAG pipeline using Bedrock Knowledge Bases with Titan Embeddings, and configured Guardrails. I haven't done this in an enterprise production environment yet, but I can speak to the architecture deeply, and given my infra background, I'm confident I can get to production quality quickly. I'm not claiming production years — I'm claiming the skills to build it."

---

### Q4. "What is your expected CTC and notice period?" ⭐ ALWAYS ASKED
**Difficulty:** Easy  
> "I'm targeting 13–15 LPA as mentioned. I'm immediately available — no notice period."

---

### Q5. "Why this role at BCG specifically?"
**Difficulty:** Easy

> "BCG's engineering practice builds platforms at scale for some of the world's most complex organizations. The fact that this role involves building an AI-powered Terraform provisioning platform — not just using AI tools, but engineering the platform itself — is exactly the kind of high-impact work I want to be doing. The combination of Bedrock, Terraform Cloud APIs, and Kubernetes is my exact skill intersection."

---

## PART 2 — Technical Screening Questions

> These will be surface-to-mid level in this IWS round. They're checking if you know enough to be credible with BCG.

---

### Q6. "What is AWS Bedrock? How is it different from SageMaker?" ⭐⭐ VERY LIKELY
**Difficulty:** Easy–Medium

**Answer:**
> "Bedrock is a fully managed service to access foundation models from providers like Anthropic, Meta, Mistral, and Amazon's own Titan and Nova — all via a single API with no infrastructure to manage. It's serverless. SageMaker is for custom model training, fine-tuning, and building full MLOps pipelines — you manage compute instances. For this JD, Bedrock is the right choice: we're consuming FMs and building RAG/agent workflows, not training custom models."

---

### Q7. "Explain what RAG is and how you'd use it in this platform." ⭐⭐ VERY LIKELY
**Difficulty:** Medium

**Answer:**
> "RAG — Retrieval-Augmented Generation — grounds LLM responses in real data rather than relying on the model's training knowledge alone. The flow is: documents are chunked, converted to vector embeddings, and indexed in a vector store. At query time, the user's input is also embedded, and a similarity search finds the most relevant chunks. Those chunks are injected into the LLM's prompt as context, so the model generates answers based on actual retrieved content, not hallucination.
>
> For this platform specifically — a user types 'create an EKS cluster with 3 node groups' — RAG retrieves relevant Terraform module documentation from our knowledge base, and the LLM generates a config using real module variable names and signatures, dramatically reducing hallucinations."

---

### Q8. "What Bedrock components have you worked with?" ⭐⭐ VERY LIKELY
**Difficulty:** Medium

**Answer:**
> "I've worked with: InvokeModel API for direct Claude calls, the Converse API for multi-turn chat, Bedrock Knowledge Bases for managed RAG — which handles chunking, embedding with Titan Embeddings, and indexing to OpenSearch Serverless. I've also explored Bedrock Guardrails to configure content filters and topic denial policies. And I've read through the Agents API — where you define Action Groups backed by Lambda functions, and the agent uses ReAct-style reasoning to decide which action to call."

---

### Q9. "Walk me through how Terraform Cloud is different from Terraform OSS." ⭐⭐ VERY LIKELY
**Difficulty:** Easy–Medium

**Answer:**
> "Open-source Terraform runs locally or in CI/CD — you manage state yourself, typically via S3 + DynamoDB. Terraform Cloud adds: centralized remote state with locking, remote plan/apply execution, a module registry, workspace management, VCS integration, and Sentinel policy-as-code enforcement between plan and apply. For this platform, TFC is essential because we need to trigger runs programmatically via API, enforce policies before apply, and maintain audit trails — all of which TFC handles natively."

---

### Q10. "How do you call the Terraform Cloud API to trigger a run?" ⭐ LIKELY
**Difficulty:** Medium

**Answer:**
> "You authenticate via a Bearer token in the Authorization header. To trigger a run, you POST to `/api/v2/runs` with the workspace ID in the request body and a message. TFC then queues a plan — you can poll the run status via GET on the run ID. If the plan succeeds and policies pass, you trigger apply either via `auto-apply` flag or another POST to `/runs/{id}/actions/apply`. I've written Python scripts for this workflow — creating workspaces, setting variables, triggering runs, and polling status."

---

### Q11. "Explain IRSA and how you'd use it for a Bedrock-calling pod on EKS." ⭐⭐ VERY LIKELY — you have production exp here
**Difficulty:** Medium  
**This is your strongest card — own it.**

**Answer:**
> "IRSA — IAM Roles for Service Accounts — lets EKS pods assume IAM roles without static credentials. The flow: create an OIDC identity provider in IAM pointing to the cluster's OIDC URL, create an IAM role with a trust policy scoped to the specific namespace and service account name, then annotate the Kubernetes service account with `eks.amazonaws.com/role-arn`. The pod gets a projected token that it exchanges for AWS credentials via the OIDC flow.
>
> For a Bedrock-calling service — I'd attach an IAM role with `bedrock:InvokeModel` and `bedrock:RetrieveAndGenerate` permissions to the pod's service account. No static keys anywhere in the codebase or secrets. I implemented IRSA in production at Protean for our EKS clusters — for eSign and eKYC services — so I'm very comfortable with this."

---

### Q12. "What is Kubernetes HPA and when would you use it for an AI service?" ⭐ LIKELY
**Difficulty:** Easy–Medium

**Answer:**
> "HPA — Horizontal Pod Autoscaler — scales pod replicas based on CPU, memory, or custom metrics. For a Bedrock API gateway service that proxies LLM calls, HPA on CPU/request count makes sense — traffic spikes during business hours would scale up pods automatically. One consideration: Bedrock itself is serverless and scales on AWS's side, but the FastAPI/Lambda wrapper around it can become a bottleneck. I'd also use VPA for right-sizing if the workload has unpredictable memory patterns."

---

### Q13. "What is Checkov and where does it fit in an AI Terraform pipeline?" ⭐ LIKELY
**Difficulty:** Easy

**Answer:**
> "Checkov is a static analysis tool for IaC — it scans Terraform files or plan JSON output for security misconfigurations before deployment. Common checks: S3 bucket public access, unrestricted security groups, unencrypted EBS, missing MFA on IAM users. In an AI-generated Terraform pipeline, Checkov is the critical guardrail — after the LLM generates HCL, you run `checkov -d ./generated_tf/` before submitting to TFC. Any failed policy blocks the submission. This prevents the AI from generating technically valid but insecure infrastructure."

---

### Q14. "How would you monitor an LLM-powered service in production?" ⭐ MEDIUM PROBABILITY
**Difficulty:** Medium

**Answer:**
> "Multiple layers: at the infrastructure level — CloudWatch for Lambda/EKS metrics, ALB latency, error rates, 5xx alarms. At the application level — log every LLM request/response with token counts, latency, and model ID to CloudWatch Logs or OpenSearch. For AI-specific monitoring: track input/output token counts for cost control, measure response latency per model, alert on guardrail trigger rates (high rate = users trying to misuse the system). I'd use Bedrock's built-in invocation logging to S3 plus CloudWatch metrics for model-level visibility."

---

## PART 3 — Scenario Questions

### Q15. "A user asks your platform to 'delete all S3 buckets in production' — what happens?" ⭐⭐ HIGH PROBABILITY
**Difficulty:** Medium–Hard

**Answer:**
> "This is exactly why the platform has multiple guardrail layers. First, the system prompt constrains the LLM — it's instructed to only handle provisioning of approved resource types, not deletions of existing infrastructure. Second, Bedrock Guardrails would have a topic denial policy blocking destructive operations. Third, even if somehow a deletion config was generated, Checkov would flag it, and Sentinel policies in TFC would hard-block any `destroy` operations on production workspaces. Fourth, the TFC workspace for production would require manual approval before apply — no auto-apply. Defense in depth — no single point of failure."

---

### Q16. "Tell me about a time you automated something complex." ⭐ ALWAYS IN HR ROUNDS
**Difficulty:** Easy — use your Protean story

**Answer:**
> "At Protean, we had quarterly DR drills across 1500+ VMware VMs — traditionally a multi-day manual process. I designed Ansible-based DR automation playbooks that scripted the full failover sequence. The result was 60% reduction in drill execution time and consistent RPO/RTO compliance across all quarterly audits. The key was mapping out every manual step the team was doing, understanding the dependencies, and encoding them as idempotent Ansible tasks with validation checks at each stage."

---

## PART 4 — Questions YOU Should Ask

Ask exactly 2 of these — shows interest and preparation:

1. **"Is this an IWS internal screening before submission to BCG's technical panel, or is this the primary technical round?"** *(Lets you calibrate the depth of this conversation vs. what's coming next)*

2. **"What does the current state of this platform look like — are you building from scratch, or enhancing an existing system?"**

3. **"What's the expected timeline for the BCG technical round if this screening goes well?"**

---

## Quick Difficulty Reference — All Questions

| # | Question | Difficulty | Your Strength |
|---|---|---|---|
| Q1 | Tell me about yourself | 🟢 Easy | Scripted answer ready |
| Q2 | Why leaving Protean? | 🟡 Medium | Honest answer ready |
| Q3 | Bedrock production exp? | 🟡 Medium | Be honest, bridge well |
| Q4 | CTC / Notice period | 🟢 Easy | Immediate, 13–15 LPA |
| Q5 | Why BCG? | 🟢 Easy | Platform + AI intersection |
| Q6 | Bedrock vs SageMaker | 🟡 Medium | Know it cold |
| Q7 | What is RAG? | 🟡 Medium | Explain with Terraform use case |
| Q8 | Bedrock components used | 🟡 Medium | InvokeModel, KB, Guardrails |
| Q9 | TFC vs Terraform OSS | 🟢 Easy | Know your Terraform |
| Q10 | TFC API trigger run | 🟡 Medium | POST /runs endpoint |
| Q11 | IRSA for Bedrock pod | 🟢 Easy for you | PRODUCTION experience — ace this |
| Q12 | HPA for AI service | 🟢 Easy | Standard K8s knowledge |
| Q13 | Checkov in pipeline | 🟢 Easy | Static IaC analysis |
| Q14 | Monitor LLM in prod | 🟡 Medium | CloudWatch + token tracking |
| Q15 | Delete S3 scenario | 🔴 Hard | Multi-layer guardrail answer |
| Q16 | Automation story | 🟢 Easy | Ansible DR story from Protean |

---

## IWS Company Context (Know Before You Walk In)

- **What IWS is:** ICT Solutions and Professional Services company, founded 2004, ~585 employees, HQ Noida. Primary business: Telecom (4G/5G/ORAN), IT Services, Cloud, Data Analytics, Managed Services.
- **Tech stack IWS uses internally:** Microsoft 365, Cloudflare CDN — mostly a services/staffing company, not a product company. For BCG delivery, the tech stack is BCG's.
- **The interview panel is HR-only** — Tejaswi, Sabrina, Meenakshi, Sumit are all HR/Resource Management. This is a fit screening, not a deep technical panel.
- **IWS → BCG relationship:** IWS places permanent employees with BCG as a staffing/solutions partner. You'll be on IWS payroll, deployed full-time to BCG.
- **What BCG cares about (next round):** Technical depth in Bedrock, RAG, Terraform Cloud integration, Python, and ability to work independently in Agile sprints.

---

## Final 30-Minute Game Plan

```
Be online 5 minutes early on Teams.
Keep your resume open on a second screen.
Have a glass of water.

Your 3 priorities for this call:
1. Sound confident and clear — not robotic, not nervous
2. Be honest about your AI experience — bridge to architecture knowledge
3. Show enthusiasm for this exact use case (AI + Terraform + Cloud)

This is a screening gate, not the final technical round.
Pass this → BCG technical panel → that's where the deep Bedrock questions come.
```

---

*Prepared for: Digambar Rajaram | Interview Date: ~June 2026 | Via: Teams (IWS Internal Screening)*
