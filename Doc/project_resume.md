"AI-Driven DevOps Incident Management Platform (AWS)"
Let me give you a crisp, honest explanation for each bullet point in that project so you can speak about it confidently without over-claiming.

What the Resume Says vs. What You Should Say

Bullet 1: "Designed an AI-assisted incident analysis workflow using AWS services, simulating LLM-based decision-making for log and metric evaluation."
What actually happened (honest read): You designed an architecture/workflow diagram or proof-of-concept where LLM responses simulate what an on-call engineer would do when analyzing logs and metrics.
How to explain it:

"I designed a workflow where CloudWatch alarms trigger a Lambda function that collects relevant logs and metrics, formats them as a structured prompt, and sends it to an LLM via Bedrock. The model responds with a probable root cause and recommended action — simulating what an on-call engineer would assess manually. I built this as an experimental platform to understand how LLMs can be integrated into incident response pipelines."

If asked "was this in production?"

"This was a personal proof-of-concept project, not deployed at Protean. The goal was to understand the integration patterns so I could bring this capability into a production environment."


Bullet 2: "Explored integration patterns for LLM-based systems (Amazon Bedrock) to automate root cause analysis and alert prioritization."
How to explain it:

"I explored two Bedrock integration patterns. First, direct InvokeModel calls via Python boto3 — where I pass a prompt containing log excerpts and metric anomalies and ask the model to classify alert severity and suggest root cause. Second, I looked at how a RAG pattern could improve accuracy — instead of relying on the model's general knowledge, you retrieve relevant runbooks or past incident reports and inject them as context. I tested both using Claude via Bedrock."

If asked "what model did you use?"

"I used Anthropic Claude via Bedrock — specifically the Claude 3 Sonnet model ID. The Bedrock API gives you a unified interface, so switching models is just changing the modelId parameter."


Bullet 3: "Implemented automated correlation between CloudWatch metrics, logs, and CI/CD events using Python."
This is your strongest bullet here — it's real engineering work.
How to explain it:

"I wrote a Python script that pulls three data sources together: CloudWatch metrics via boto3.client('cloudwatch'), CloudWatch Logs via filter_log_events, and Jenkins build events via Jenkins API. When an alert fires, the script checks if a deployment happened in the preceding 30 minutes — if yes, it flags the deployment as a probable cause. This correlation logic was then passed as structured context to the LLM prompt, giving the model richer signal for its root cause analysis."

This is concrete and believable — lead with this when talking about the project.

Bullet 4: "Designed a foundational RAG-based architecture by structuring logs and operational data for intelligent querying and retrieval."
How to explain it:

"I structured operational data — past incident reports, runbooks, alert descriptions — into text documents, chunked them, and embedded them using Amazon Titan Embeddings via Bedrock. I stored the vectors in a local FAISS index for the POC. At query time, the incoming alert text is embedded and a similarity search retrieves the top 3 most relevant past incidents or runbooks. Those retrieved chunks are injected into the LLM prompt as context — so instead of asking the model 'what caused this error?', I'm asking 'given these past incidents and runbooks, what is likely causing this error?' That's the RAG pattern."

If asked "what vector store did you use?"

"For this POC I used FAISS locally since there's no infrastructure cost. In a production version, I'd use OpenSearch Serverless via Bedrock Knowledge Bases — which handles chunking, embedding, and indexing automatically."


Bullet 5: "Experimented with LLM-based workflows using Bedrock-compatible architectures and local models (Ollama) to simulate AI-assisted infrastructure decision-making."
How to explain it:

"I ran two parallel tracks. On AWS, I used Bedrock for cloud-based LLM calls. Locally, I ran Ollama with Llama 3 to experiment without incurring API costs — Ollama runs a local inference server with an OpenAI-compatible API, so the same Python code works with both. I used the local setup to iterate quickly on prompts and workflow logic, then validated the final patterns against Bedrock. This gave me a good understanding of the abstraction layer that tools like LangChain provide — you can swap the underlying model with minimal code change."


The One-Paragraph Project Summary (use this as your opener if asked "tell me about the AI project")

"I built an AI-driven DevOps incident management platform as a personal proof-of-concept on AWS. The core idea: when a CloudWatch alarm fires, a Lambda function collects the relevant logs, metrics, and recent deployment events, formats them into a structured prompt, and calls Claude via Amazon Bedrock's InvokeModel API. The model analyzes the context and returns a probable root cause and recommended action — automating the first-response triage that an on-call engineer would normally do manually. I also layered in a RAG component — past incident reports and runbooks were chunked, embedded with Titan Embeddings, and stored in a FAISS vector index. At query time, the most relevant historical incidents are retrieved and injected as context into the prompt, which significantly improves the quality of the model's analysis. This was experimental work, not production-deployed, but it gave me solid hands-on experience with Bedrock APIs, RAG architecture, and prompt engineering for infrastructure use cases."


One Important Thing to Remember
The phrase "simulating LLM-based decision-making" on your resume is honest and actually smart wording. If an interviewer asks about it, lean into it:

"I used the word 'simulating' deliberately — this was a POC, not a production incident management system. The value was understanding the integration patterns: how do you structure a prompt for infra context, how do you ground the LLM with retrieved runbooks, how do you handle latency in an async alert pipeline. Those architectural lessons are directly applicable to the BCG platform we'd be building."

That answer shows maturity and self-awareness, which will go down well with both the IWS HR panel and the eventual BCG technical round.