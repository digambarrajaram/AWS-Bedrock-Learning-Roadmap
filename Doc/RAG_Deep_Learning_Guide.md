# Retrieval-Augmented Generation (RAG) with Amazon Bedrock Knowledge Bases
## A Comprehensive, Deep-Dive Learning Guide

---

**Document Version:** 1.0  
**Last Updated:** June 2025  
**Target Audience:** Cloud AI Engineers, DevOps professionals transitioning to AI infrastructure, ML Engineers, and anyone preparing for roles requiring AWS Bedrock + RAG expertise.

---

# Part 0: Introduction & Prerequisites

## 0.1 What This Guide Covers

This guide is a definitive, self-contained reference on Retrieval-Augmented Generation (RAG) — from foundational theory to production-grade implementation using Amazon Bedrock Knowledge Bases. It is designed to take you from zero knowledge of RAG to being able to architect, build, evaluate, and operate production RAG systems.

The guide is organized into 18 parts plus appendices:

| Part | Topic | Depth |
|------|-------|-------|
| 0 | Introduction & Prerequisites | Foundational |
| 1 | The Problem RAG Solves | Conceptual |
| 2 | Vector Embeddings — Mathematical Foundation | Deep |
| 3 | Document Processing & Chunking | Very Deep |
| 4 | Vector Databases | Comprehensive |
| 5 | The RAG Pipeline — Step by Step | Deep |
| 6 | Retrieval Strategies | Deep |
| 7 | Context Assembly & Prompt Engineering | Deep |
| 8 | Amazon Bedrock Knowledge Bases | **Very Deep** |
| 9 | Amazon Titan Embeddings | Deep |
| 10 | Building RAG with AWS — Code Examples | Practical |
| 11 | Advanced RAG Patterns | Advanced |
| 12 | Evaluation & Metrics | Deep |
| 13 | Production Operations | Deep |
| 14 | Security, Privacy, Compliance | Deep |
| 15 | Troubleshooting Guide | Practical |
| 16 | Interview Q&A | 50+ Questions |
| 17 | Hands-On Labs | 5 Labs |
| 18 | Appendices | Reference |

## 0.2 Prerequisites

Before diving in, you should have:

- **Basic Python familiarity**: You will read and write Python code using `boto3`, `langchain`, and related libraries.
- **Fundamental AWS knowledge**: Understanding of S3, IAM, CloudWatch, and basic cloud primitives.
- **Basic ML/AI familiarity**: What a language model is, what embeddings mean at a high level.
- **Familiarity with REST APIs**: How to make API calls, read API documentation.

You do **not** need to be a deep learning expert, a vector math PhD, or an experienced MLOps engineer. This guide builds from first principles.

## 0.3 How to Use This Guide

### Study Path

If you are new to RAG, follow this path:

1. **Weeks 1-2**: Read Parts 0–4 (foundations: embeddings, chunking, vector DBs)
2. **Weeks 3-4**: Read Parts 5–7 (RAG pipeline, retrieval, prompt engineering)
3. **Weeks 4-6**: Deep-dive Parts 8–9 (Bedrock Knowledge Bases + Titan Embeddings)
4. **Weeks 6-8**: Complete Parts 10–11 (code examples + advanced patterns)
5. **Weeks 8-10**: Parts 12–15 (evaluation, production ops, security, troubleshooting)
6. **Week 11+**: Parts 16–18 (interview prep + hands-on labs)

### Reference Path

If you need to look up specific topics:

- **RAG architecture questions** → Part 5
- **Bedrock KB API specifics** → Part 8, Section 8.8
- **Chunking strategies** → Part 3, Section 3.11
- **Troubleshooting** → Part 15
- **Interview prep** → Part 16

---

# Part 1: The Problem RAG Solves

## 1.1 LLM Limitations: The Big Three

Large Language Models (LLMs) are extraordinarily capable, but they have three fundamental limitations that RAG directly addresses.

### 1.1.1 Hallucination

LLMs generate text by predicting the next token based on patterns learned during training. They do not "know" facts — they generate plausible-sounding text. When asked about information not in their training data, or about niche topics where their training data was thin, they confidently produce incorrect answers. This is called **hallucination**.

Example:
```
User: "What does AWS Infrastructure Edge Locations policy say about 
       data residency for healthcare customers in the EU as of Q3 2024?"
LLM: [Confidently generates a plausible but incorrect policy summary]
```

No LLM, regardless of size, can reliably answer this because it depends on:
- A specific document (the policy) that may have changed after training
- A specific timeframe (Q3 2024)
- A specific region and industry combination

### 1.1.2 Stale Knowledge

LLMs have a **knowledge cutoff date**. They cannot know:
- Internal company policies written last week
- Customer records updated in real-time
- Breaking news from this morning
- Product documentation for a new release

For a company using an LLM to answer questions about their own products and policies, stale knowledge is a disqualifying limitation.

### 1.1.3 No Access to Private Data

LLMs trained on public internet data cannot access:
- Internal wikis and Confluence spaces
- Private repositories of contracts, legal documents
- Proprietary research and development notes
- HR policies, onboarding documents
- Customer support histories

This is both a security/privacy issue and a practical one: the most valuable knowledge in any organization is private.

## 1.2 Fine-Tuning vs. RAG: When to Use Which

A common question is: "Should I fine-tune my model or use RAG?" The answer depends on what you are trying to achieve.

### Fine-Tuning

Fine-tuning is the process of further training a pre-trained LLM on a specific dataset to adjust its weights. This changes the model's behavior — its writing style, its domain knowledge, how it structures responses.

**Fine-tuning is good for:**
- Teaching the model a specific writing style or tone
- Adapting the model to a specific domain's terminology and reasoning patterns
- Reducing token usage by having the model "know" things implicitly (smaller prompts)
- Cases where you need the model to reason consistently in a specific domain

**Fine-tuning is NOT good for:**
- Providing up-to-date factual information (you'd still need to fine-tune on the latest data, repeatedly)
- Answering questions about specific documents
- Giving the model knowledge that changes frequently
- Maintaining citations and attribution
- Keeping private data out of the model's weights (fine-tuning on private data bakes it into the model)

### RAG

RAG retrieves relevant documents at query time and includes them in the prompt. The model generates answers based on the retrieved content.

**RAG is good for:**
- Answering questions about specific documents (policies, contracts, code)
- Keeping knowledge current (update the document store, not the model)
- Providing citations and attribution
- Maintaining data security (documents stay in your infrastructure)
- Knowledge that changes frequently
- Multi-domain knowledge (a single KB can span many topics)

**RAG is not ideal for:**
- Changing the model's core personality or writing style
- Teaching the model complex reasoning patterns unique to a domain
- Scenarios where retrieval latency is unacceptable

### Decision Matrix

| Factor | Favor Fine-Tuning | Favor RAG |
|--------|------------------|-----------|
| Knowledge changes frequently | No | Yes |
| Need citations/attribution | No | Yes |
| Private data that must not leave infrastructure | No | Yes |
| Need consistent domain terminology | Yes | Partial |
| Writing style adaptation | Yes | No |
| Large knowledge base (>100K docs) | No | Yes |
| Very large number of unique entities | No | Yes |
| Model needs to "just know" something implicitly | Yes | No |

> **Practical Note:** In production systems, fine-tuning and RAG are often used together. Fine-tune for style and reasoning patterns; use RAG for factual recall with citations.

## 1.3 The RAG Intuition: Give the Model a Reference Book

Think of RAG as giving the model a reference book at query time.

Without RAG, asking an LLM about your company's expense policy is like asking a well-read human who has never read your specific policy document to guess what it says. They might be close, but they'll get details wrong.

With RAG, you retrieve the actual expense policy document and include it in the prompt. The model now has the ground truth in context. It's like handing the human the actual document and asking them to answer based on it.

```
┌──────────────────────────────────────────────────────────────────┐
│  WITHOUT RAG                                                       │
│  User Query ──────► LLM ──────► [Possibly hallucinated] Answer   │
│                                                                     │
│  WITH RAG                                                          │
│  User Query ──────► Retriever ──────► [Relevant Docs]             │
│                         │                                         │
│                         ▼                                         │
│                    LLM + Docs ──────► Grounded Answer + Citations  │
└──────────────────────────────────────────────────────────────────┘
```

This simple intuition — retrieve relevant context, include it in the prompt — is the core of RAG. The complexity comes from doing retrieval well, at scale, with low latency, and at reasonable cost.

## 1.4 Real-World Use Cases

### 1.4.1 Internal Documentation Q&A

Employees asking questions about HR policies, engineering runbooks, architecture decision records (ADRs), and onboarding guides. The system retrieves the most relevant internal documents and generates answers with citations pointing to the source.

### 1.4.2 Code Generation and Understanding

Developers asking questions about a large codebase. The RAG system retrieves relevant code files, API documentation, and architecture diagrams. The model can then explain how code works, suggest changes, or generate new code consistent with the existing architecture.

### 1.4.3 Legal Document Analysis

Lawyers and compliance teams analyzing contracts, regulatory filings, and legal precedents. The system retrieves relevant case law, prior contracts with similar clauses, and regulatory guidance, enabling faster and more thorough analysis.

### 1.4.4 Medical Information Retrieval

Healthcare professionals accessing medical literature, drug interaction databases, and clinical guidelines. The system retrieves relevant clinical studies and guidelines to support evidence-based decision-making.

### 1.4.5 Financial Research

Analysts querying earnings reports, SEC filings, analyst reports, and news articles. The system retrieves relevant financial documents and news to support investment research and reporting.

---

# Part 2: Vector Embeddings — The Mathematical Foundation

## 2.1 What Is an Embedding? Geometric Intuition

An **embedding** is a dense numerical representation of data (text, images, audio, code) as a vector of floating-point numbers in a high-dimensional space.

The key intuition: **similar items are close to each other in this space**.

If the sentence "How do I reset my password?" is embedded as a vector, it will be close in vector space to "I forgot my login credentials" and far from "How do I generate an AWS access key?" This property — **semantic similarity in vector space ≈ semantic similarity in meaning** — is what makes embeddings the backbone of RAG retrieval.

Embeddings typically have 256 to 4096 dimensions. A sentence like "Hello, world!" might be embedded as:

```
[0.0231, -0.0412, 0.0893, ..., -0.0127]  // 1536-dimensional vector
```

Modern embedding models normalize most outputs to unit length (L2 norm = 1.0), which simplifies similarity calculations.

## 2.2 How Text Becomes a Vector: The Embedding Process

Text embedding is a multi-step process:

### Step 1: Tokenization
The text is split into tokens (sub-word units). Different models use different tokenizers:
- **OpenAI's cl100k_base**: ~1 token per 4 characters of English text
- **Titan's tokenizer**: Custom, optimized for multilingual text
- **BPE-based tokenizers**: Learn token boundaries from training data

### Step 2: Neural Network Forward Pass
The token IDs are passed through a deep neural network (typically a Transformer encoder). Each token gets a contextualized representation based on its position and surrounding tokens.

### Step 3: Pooling
The per-token representations are pooled into a single vector. Common pooling strategies:
- **CLS pooling**: Use the representation of the `[CLS]` (classification) token
- **Mean pooling**: Average all token representations
- **Max pooling**: Take the max value across each dimension

### Step 4: Projection (Optional)
The pooled vector may be projected to the target embedding dimension (e.g., 1536 for Titan Embeddings v2).

### Step 5: Normalization
The final vector is L2-normalized so that the Euclidean length equals 1.0. This ensures that cosine similarity equals simple dot product, which is computationally simpler.

## 2.3 Dense vs. Sparse Vectors

### 2.3.1 Dense Vectors

Most embedding models produce **dense vectors** — most or all dimensions have non-zero values.

```
[0.023, -0.041, 0.089, 0.012, -0.077, 0.034, ..., -0.013]
```

**Characteristics:**
- Dimensions are not human-interpretable
- Information is distributed across all dimensions
- Efficient storage (fixed size, compressed via normalization)
- Most popular for RAG applications

### 2.3.2 Sparse Vectors

Sparse vectors have mostly zero values, with non-zero values at specific indices. BM25 and TF-IDF produce sparse vectors.

Example: TF-IDF representation of "cat sat on mat":
```
[0, 0, 0, 0.8, 0, 0, 0.3, 0, 0, ..., 0.5, 0, ...]
              cat        mat                      sat
```

**Characteristics:**
- Non-zero dimensions often correspond to specific terms (more interpretable)
- High dimensionality equals vocabulary size (10,000-100,000+)
- Better for exact keyword matching
- Used in hybrid search alongside dense vectors

## 2.4 Embedding Dimensions and What They Mean

The number of dimensions in an embedding vector determines:
- **Expressiveness**: More dimensions can capture more nuanced relationships
- **Storage size**: 768-dim vectors are twice as large as 384-dim vectors
- **Compute cost**: Similarity calculations are O(d) where d = dimensions
- **Risk of overfitting**: Very high dimensions with insufficient data

| Model | Dimensions | Typical Use Case |
|-------|------------|-----------------|
| Titan Embeddings v2 | 1536 | AWS-native RAG |
| Titan Embeddings v1 | 4096 | Legacy AWS applications |
| text-embedding-3-small (OpenAI) | 1536 | Cost-efficient general use |
| text-embedding-3-large (OpenAI) | 3072 | High-quality general use |
| Cohere embed-english-v3.0 | 1024 | English-dominant workloads |
| BGE-large-en-v1.5 | 1024 | Open-source, high quality |
| MiniLM | 384 | Low-latency, edge deployment |

## 2.5 Similarity Metrics: Mathematical Foundation

Similarity between two vectors measures how "close" they are in the embedding space. The choice of metric affects both retrieval quality and performance.

### 2.5.1 Cosine Similarity

The cosine of the angle between two vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` is the dot product (sum of element-wise products)
- `||A||` and `||B||` are the L2 norms

**Range:** -1 to +1 (for normalized vectors: 0 to +1)

**Intuition:** Measures the orientation, not magnitude. Two vectors pointing in the same direction have cosine similarity = 1.0, regardless of their lengths.

**When to use:** Most common choice for text embeddings. Works well when you care about semantic direction rather than magnitude. Default for most vector databases and embedding models.

### 2.5.2 Dot Product (Unnormalized)

```
dot_product(A, B) = Σ(Ai × Bi)
```

**Range:** -∞ to +∞ (theoretically), but typically limited in practice.

**When to use:** When vectors are L2-normalized, dot product equals cosine similarity and is computationally simpler (no division). Many production systems use this because normalization happens at embedding time.

### 2.5.3 Euclidean Distance (L2 Distance)

```
euclidean(A, B) = ||A - B|| = sqrt(Σ(Ai - Bi)²)
```

**Range:** 0 to +∞

**Intuition:** Straight-line distance in geometric space. Good when magnitude matters.

**When to use:** When you care about both direction AND magnitude. Useful in recommendation systems where popularity (magnitude) matters.

### 2.5.4 Relationship Between Metrics

For L2-normalized vectors:
```
cosine_similarity(A, B) = dot_product(A, B)
euclidean(A, B) = sqrt(2 × (1 - cosine_similarity(A, B)))
```

So for normalized vectors, cosine similarity and Euclidean distance are monotonically related — maximizing cosine is equivalent to minimizing Euclidean distance.

## 2.6 Vector Space Visualization Intuition

Imagine a 2D representation of embeddings for these sentences:

```
                        [far] "How do I bake a chocolate cake?"
                              /
                             /
[medium] "Recipe for chocolate cake"        "How do I reset my AWS root account?"
         /
        /
[close] "I need the password reset procedure"
              \
               \
[very close] "How do I reset my password?"
```

In this simplified 2D view:
- "How do I reset my password?" and "I need the password reset procedure" are close (same intent)
- They are far from "How do I bake a chocolate cake?" (different topic)
- "Recipe for chocolate cake" is somewhat between the two (shares the word "cake")

In real 1024+ dimensional space, these relationships are far richer, capturing nuance like:
- Sentiment ("great" vs "terrible" embeddings are far apart even with similar words)
- formality ("ascending" vs "going up")
- domain ("AWS IAM" vs "military rank")

## 2.7 Common Embedding Models

### 2.7.1 Amazon Titan Embeddings

Amazon's own embedding models, deeply integrated with Bedrock:

| Model | Dimensions | Max Input | Languages | Normalized |
|-------|-----------|-----------|-----------|------------|
| Titan Embeddings G1 - Text v2 | 1536 | 8K tokens | 25+ | Yes |
| Titan Embeddings G1 - Text v1 | 4096 | 8K tokens | 25+ | Yes |
| Titan Multimodal | 1024 | Text + Image | 25+ | Yes |

**Key features:**
- Optimized for AWS infrastructure (low-latency inference within Bedrock)
- Integrated with Bedrock Knowledge Bases (no data leaves AWS)
- Supports semantic search and text classification
- Dimension-reducible without retraining (Matryoshka-style truncation)

### 2.7.2 OpenAI Embeddings

| Model | Dimensions | Max Input | Training Data |
|-------|-----------|-----------|---------------|
| text-embedding-3-large | 3072 | 8K tokens | Up to Sep 2021 |
| text-embedding-3-small | 1536 | 8K tokens | Up to Sep 2021 |
| text-embedding-ada-002 | 1536 | 8K tokens | Legacy |

**Note:** OpenAI embeddings have a knowledge cutoff — they do not know about events after training.

### 2.7.3 Cohere Embed

| Model | Dimensions | Max Input | Multilingual |
|-------|-----------|-----------|-------------|
| embed-english-v3.0 | 1024 | 512 tokens | English only |
| embed-multilingual-v3.0 | 1024 | 512 tokens | 100+ languages |
| embed-english-light-v3.0 | 384 | 512 tokens | English only |

### 2.7.4 Open-Source Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| BGE-large-en-v1.5 | 1024 | Best open-source English model |
| BGE-m3 | 1024 | Multilingual, supports 100+ languages |
| E5-Mistral-7B | 4096 | High quality, requires more compute |
| MiniLM-L12 | 384 | Fast, low memory, good quality |

## 2.8 Multi-Modal Embeddings

Modern embedding models can embed multiple modalities — text, images, and sometimes audio or video — into the same vector space. This enables cross-modal retrieval.

Example: "Find the image of a red sports car" can retrieve from a database of images using a text query, because images and text are embedded in the same space.

**Multi-modal models for RAG:**
- **Titan Multimodal**: Embeds images and text for RAG with documents containing figures, charts, and diagrams
- **OpenAI CLIP**: Encodes images and text into the same 512-dim or 768-dim space
- **BGE-multimodal**: Open-source, Chinese-original but multilingual capable

## 2.9 Embedding Quality Factors

Not all embeddings are equal. Key quality factors:

### 2.9.1 Semantic Coherence
Do similar meanings map to nearby vectors? Test by computing similarity between known similar sentence pairs and checking that scores are high.

### 2.9.2 Discrimination Power
Can the model distinguish between subtly different meanings? "The bank is by the river" vs "The bank declined the loan" should have low similarity.

### 2.9.3 Length Invariance
Longer documents should not automatically have higher or lower similarity scores just because of length. Good models handle this through attention mechanisms.

### 2.9.4 Multilingual Alignment
For multilingual corpora, semantically equivalent sentences in different languages should have high similarity.

### 2.9.5 Benchmark Datasets

Common benchmarks for embedding quality:

| Benchmark | What It Tests |
|-----------|--------------|
| MTEB | Massive Text Embedding Benchmark (56 datasets) |
| BEIR | Benchmark for Information Retrieval |
| STS-B | Semantic Textual Similarity Benchmark |
| MS MARCO | Passage ranking on Bing queries |
| ArguAna | Argument retrieval |

---

# Part 3: Document Processing & Chunking

Chunking — the process of splitting documents into smaller, semantically coherent pieces — is one of the most impactful and underrated decisions in RAG system design. Poor chunking can cripple retrieval quality even with a perfect embedding model and vector database.

## 3.1 Why Chunking Matters

### 3.1.1 Context Window Constraints

LLMs have a finite context window (e.g., 128K tokens for Claude 3.5 Sonnet, 200K for Gemini 1.5). If you chunk documents into 10K-token pieces, only a few chunks can fit in context. If you chunk into 50-token pieces, you get more candidates but less context per chunk.

The goal: chunks should be small enough to fit many in context, but large enough to be self-contained.

### 3.1.2 Relevance Granularity

Users ask questions at a specific granularity. A question like "What is the redaction policy for PII?" should retrieve the relevant section of a privacy policy, not the entire 200-page document, and not a single sentence.

Chunk size determines the granularity of retrieval. Too large = includes irrelevant surrounding context. Too small = missing surrounding context needed for understanding.

### 3.1.3 Embedding Quality

Embedding models have context windows (e.g., 8K tokens for Titan v2). Sending 8K tokens as a single chunk produces one embedding. Sending 100 80-token chunks produces 100 embeddings. The latter provides much more granular retrieval.

However, if you split a paragraph into 10 chunks, each chunk loses the context of surrounding sentences, potentially breaking semantic coherence.

### 3.1.4 Cost

Embedding generation has a per-token cost. Smaller chunks → more embeddings → higher embedding costs. But larger context → higher LLM inference costs. There is an optimal trade-off point.

## 3.2 Fixed-Size Chunking

The simplest approach: split text into chunks of N tokens or N characters, ignoring semantic boundaries.

```python
def fixed_size_chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Simple fixed-size chunking by character count.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap  # Move forward with overlap
    return chunks
```

**Pros:**
- Simple to implement
- Predictable chunk sizes and counts
- Fast processing

**Cons:**
- Breaks semantic units mid-sentence, mid-paragraph, mid-thought
- Creates incoherent chunks that lack surrounding context
- Poor retrieval quality in practice

**When to use:** Quick prototyping only. Replace with semantic chunking before production.

## 3.3 Recursive Character Text Splitting

A better approach that respects text structure hierarchy: try to split on paragraphs first, then sentences, then words, then individual characters. This preserves semantic coherence as much as possible.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],  # Try splitting at paragraph, then sentence, etc.
    chunk_size=500,
    chunk_overlap=50,
    length_function=lambda text: len(text.split()),  # Count words, not chars
)

chunks = text_splitter.split_text(document)
```

The separators list is processed in order: try splitting on `"\n\n"` first; if a chunk is still too large, try `"\n"`; then `. `; then ` `; finally fall back to individual characters.

**Pros:**
- Respects semantic boundaries (paragraphs > sentences > words)
- Much better coherence than fixed-size
- Configurable separators for different document types

**Cons:**
- Chunk sizes still have some variance
- May not handle structured documents (tables, code) well
- Does not understand actual semantic content

## 3.4 Semantic Chunking

Semantic chunking uses the embedding model to identify natural breaking points where content shifts in meaning. Two common approaches:

### 3.4.1 Embedding-Based Semantic Chunking

```python
import numpy as np

def semantic_chunk_by_embedding(
    text: str,
    splitter,
    embedding_model,
    threshold: float = 0.5,
    min_chunk_size: int = 100
) -> list[str]:
    """
    Chunk text by detecting semantic shifts using embedding similarity.
    
    Algorithm:
    1. Split text into candidate segments (sentences or paragraphs)
    2. Compute embeddings for each segment
    3. Merge consecutive segments where similarity > threshold
    4. Ensure chunks meet minimum size
    """
    # Split into basic units (sentences via splitter)
    segments = splitter.split_text(text)
    
    if len(segments) == 0:
        return []
    
    # Compute embeddings for all segments
    embeddings = embedding_model.embed_documents(segments)
    
    chunks = []
    current_chunk = segments[0]
    current_embedding = embeddings[0]
    
    for i in range(1, len(segments)):
        # Compute similarity between current chunk and new segment
        similarity = cosine_similarity(current_embedding, embeddings[i])
        
        if similarity > threshold and len(current_chunk) + len(segments[i]) < 2000:
            # Merge: add to current chunk and recompute embedding (approximation)
            current_chunk += " " + segments[i]
            # In production, you'd recompute the embedding here
        else:
            # Start a new chunk
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
            current_chunk = segments[i]
            current_embedding = embeddings[i]
    
    # Don't forget the last chunk
    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk)
    
    return chunks
```

### 3.4.2 LLM-Based Semantic Chunking

Use an LLM to identify natural topic boundaries:

```python
def llm_semantic_chunker(document: str, target_chunk_size: int = 500) -> list[dict]:
    """
    Use an LLM to identify semantic boundaries and create chunks.
    
    Returns chunks with metadata about why they were separated.
    """
    prompt = f"""
    Analyze the following document and identify semantic sections that should be 
    chunked together. Look for topic shifts, section breaks, and natural divisions.
    
    Target chunk size: approximately {target_chunk_size} words.
    
    For each chunk, provide:
    - The text content
    - A brief description of what this chunk covers
    - Why it was separated from adjacent chunks
    
    Return your response as a JSON array of chunks.
    
    Document:
    {document}
    """
    
    response = llm.invoke(prompt)
    # Parse response into structured chunks
    return parse_chunks_from_response(response)
```

**Pros:**
- Respects actual semantic boundaries
- Creates more coherent chunks
- Better for complex, multi-topic documents

**Cons:**
- Slower and more expensive (requires LLM calls or many embedding calls)
- Threshold tuning can be tricky
- May create uneven chunk sizes

## 3.5 Hierarchical Chunking: Parent-Child Relationships

A powerful pattern where small child chunks reference and roll up to larger parent chunks:

```
┌─────────────────────────────────────────────────────────────────┐
│ PARENT DOCUMENT: Full 2000-word technical specification         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ CHILD CHUNK  │  │ CHILD CHUNK  │  │ CHILD CHUNK  │  ...     │
│  │ (500 words)  │  │ (500 words)  │  │ (500 words)  │          │
│  │ Section 3.1  │  │ Section 3.2  │  │ Section 3.3  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│                    Retrieval                                     │
│                          │                                       │
│              ┌───────────▼────────────┐                          │
│              │ Retrieve top-k chunks  │                           │
│              │ (children)             │                           │
│              └───────────┬────────────┘                          │
│                          │                                       │
│              ┌───────────▼────────────┐                          │
│              │ Return parent docs of  │                          │
│              │ retrieved children     │                          │
│              └────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**Why:** Child chunks provide granular retrieval. Parent documents provide full context for generation. This handles queries that need understanding of a broader section.

```python
# Parent Document Retriever (LangChain pattern)
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorStore

# Child splitter (small chunks for retrieval)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# Parent splitter (large chunks for context)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n"]
)

# Vector store for child chunk embeddings
vectorstore = OpenSearchVectorStore(embedding=bedrock_embeddings)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=SimpleDocumentStore(),  # Stores parent documents
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)
```

## 3.6 Agentic Chunking (Emerging)

Agentic chunking uses an LLM agent to make intelligent decisions about chunk boundaries dynamically, based on content understanding:

1. The agent previews the document to understand its structure
2. It identifies logical sections and their relationships
3. It makes decisions about chunk boundaries that preserve meaning
4. It may create different chunking strategies for different document types within the same corpus

This is more expensive but produces the highest quality chunks for complex, heterogeneous document collections.

## 3.7 Chunk Overlap: Why and How Much

Chunk overlap includes a portion of the previous chunk in the current chunk to prevent context loss at boundaries:

```
Chunk 1: [==========500 chars==========]
Chunk 2: [---overlap---][======500 chars======]
Chunk 3: [---overlap---][======500 chars======]
```

**Why overlap helps:** A concept explained at the end of Chunk 1 might be needed at the start of Chunk 2. Overlap ensures the concept isn't split across chunks with nothing bridging it.

**How much overlap:**
- **Too little** (0 overlap): Lose context at boundaries
- **Too much** (50%+): Wasteful, redundant embeddings
- **Typical range**: 10-20% of chunk size

For a 500-token chunk, 50-100 token overlap is typical. For a 2000-character chunk, 200-400 character overlap.

**Mathematical intuition:** If concepts are distributed uniformly and chunk size is S with overlap O, the effective stride is S - O. For complete coverage with no gaps, O must be at least the maximum concept span. In practice, 10-20% works well because most concepts are shorter than 10-20% of typical chunk sizes.

## 3.8 Chunk Size Selection Heuristics

Choosing the right chunk size is domain and use-case dependent:

| Use Case | Recommended Chunk Size | Rationale |
|----------|----------------------|-----------|
| Legal contracts | 500-1000 tokens | Need clause-level granularity; clauses are self-contained |
| Technical documentation | 500-1500 tokens | Balance detail with context; sections are meaningful units |
| Code repositories | Function/class level (variable) | Natural boundaries at function level; don't split functions |
| Support tickets | 200-500 tokens | Shorter, more specific; users ask focused questions |
| Research papers | 500-1000 tokens | Abstract, methods, results each have distinct contexts |
| Long-form articles | 1000-2000 tokens | Paragraph-level chunks preserve coherence |

### Empirical Selection Method

```python
def evaluate_chunk_size(
    corpus: list[str],
    embedding_model,
    sample_queries: list[str],
    chunk_sizes: list[int] = [256, 512, 1024, 2048]
) -> dict[int, float]:
    """
    Empirically evaluate different chunk sizes using recall@k.
    
    For each chunk size, chunk the corpus, index it, and measure
    how often the relevant document appears in top-k results.
    """
    results = {}
    
    for size in chunk_sizes:
        chunks = [recursive_chunk(doc, size) for doc in corpus]
        index = build_index(chunks, embedding_model)
        
        recall_scores = []
        for query, relevant_doc in sample_queries:
            retrieved = index.query(query, k=10)
            if relevant_doc in retrieved:
                recall_scores.append(1.0)
            else:
                recall_scores.append(0.0)
        
        results[size] = np.mean(recall_scores)
    
    return results  # Pick the size with highest recall
```

## 3.9 Metadata Extraction and Enrichment

Attaching metadata to chunks dramatically improves retrieval quality by enabling post-filtering and boosting:

```python
@dataclass
class ChunkWithMetadata:
    chunk_id: str
    content: str
    document_id: str
    document_title: str
    document_type: str  # "policy", "contract", "code", "manual"
    created_date: datetime
    author: str
    version: str
    section_heading: str
    word_count: int
    language: str
    tags: list[str]
    permissions: str  # "internal", "confidential", "public"

# Extraction example
def extract_metadata(document: Document, chunk: str) -> ChunkWithMetadata:
    return ChunkWithMetadata(
        chunk_id=generate_uuid(),
        content=chunk,
        document_id=document.id,
        document_title=document.title,
        document_type=classify_document_type(document),
        created_date=document.created,
        author=document.author,
        version=document.version,
        section_heading=detect_section_heading(chunk, document),
        word_count=len(chunk.split()),
        language=detect_language(chunk),
        tags=extract_entities(chunk),  # Named entities, key topics
        permissions=document.sensitivity,
    )
```

**Post-retrieval filtering examples:**
- "Show me only contracts from 2023"
- "Answer using only unclassified documents"
- "Prioritize the latest version of each document"

## 3.10 Document Preprocessing

Raw documents arrive in many formats. Preprocessing is critical for chunk quality.

### 3.10.1 PDF Extraction

```python
import pypdf
from io import BytesIO

def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text and metadata from PDF with page-level tracking.
    
    Returns list of {"page_num": int, "text": str, "metadata": dict}
    """
    reader = pypdf.PdfReader(BytesIO(pdf_bytes))
    pages = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        metadata = {
            "page_num": i + 1,
            "total_pages": len(reader.pages),
            "page_width": page.mediabox.width,
            "page_height": page.mediabox.height,
        }
        pages.append({"page_num": i + 1, "text": text, "metadata": metadata})
    
    return pages

# For image-heavy PDFs, use document AI or OCR
# Amazon Textract can extract structured data from scanned documents
import boto3
textract = boto3.client('textract')

def extract_from_image_pdf(pdf_bytes: bytes) -> str:
    response = textract.detect_document_text(
        Document={'Bytes': pdf_bytes}
    )
    return " ".join([item['Text'] for item in response['Blocks']])
```

### 3.10.2 HTML Parsing

```python
from bs4 import BeautifulSoup
import re

def extract_text_from_html(html: str, preserve_structure: bool = True) -> str:
    """
    Extract clean text from HTML, optionally preserving structure.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script, style, nav elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    if preserve_structure:
        # Convert headings to newlines (creates semantic boundaries)
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.insert_before('\n\n## ')
            tag.insert_after('\n')
        
        # Convert lists
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                li.insert_before('• ')
        
        text = soup.get_text(separator=' ')
    else:
        text = soup.get_text()
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### 3.10.3 Table Handling

Tables are notoriously difficult to chunk correctly. A table row split across chunks is meaningless.

**Strategy 1: Serialize tables as markdown**
```python
def table_to_markdown(table_element) -> str:
    """Convert HTML table to markdown for embedding."""
    rows = table_element.find_all('tr')
    headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]
    
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(['th', 'td'])]
        md += "| " + " | ".join(cells) + " |\n"
    
    return md
```

**Strategy 2: Treat tables as atomic chunks**
Never split a table. If a table is too large, serialize it and treat it as one (potentially large) chunk.

**Strategy 3: Extract table summaries**
For very large tables, generate a summary and embed the summary rather than the full table. Store the full table for retrieval but not embedding.

### 3.10.4 Code-Specific Preprocessing

Code requires special handling:

```python
def preprocess_code(code: str, language: str) -> str:
    """
    Preprocess code to improve embedding quality.
    - Normalize whitespace
    - Remove comments (noise for semantic search)
    - Preserve function/class names (important signals)
    """
    # Normalize indentation
    lines = code.split('\n')
    # Detect and normalize leading whitespace
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    normalized = '\n'.join(
        line[min_indent:] if line.strip() else ''
        for line in lines
    )
    
    # Optionally remove comments for semantic search
    # (But keep for "find me code with this comment" searches)
    
    return normalized

# For LangChain code splitting
from langchain.text_splitter import Language

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
)

chunks = code_splitter.split_text(code_file_content)
```

## 3.11 Bedrock Knowledge Base Chunking Strategies In Detail

Amazon Bedrock Knowledge Bases provides three built-in chunking strategies plus custom options.

### 3.11.1 Strategy 1: No Chunking (Single Document)

```python
# Bedrock KB no chunking configuration
kb_config = {
    "chunkingConfiguration": {
        "chunkingStrategy": "NONE",  # Entire document as one chunk
        # Only practical for documents < 8192 tokens (Titan context)
    }
}
```

**When to use:** Short documents (policy documents, single-page specs). Rarely appropriate for large documents.

### 3.11.2 Strategy 2: Fixed-Size Chunking

```python
kb_config = {
    "chunkingConfiguration": {
        "chunkingStrategy": "FIXED_SIZE",
        "fixedSizeChunkingConfiguration": {
            "maxTokens": 500,          # Target tokens per chunk
            "overlapPercentage": 20,   # 20% overlap
        }
    }
}
```

**Parameters:**
- `maxTokens`: 100-8192, typically 300-1000 for Q&A workloads
- `overlapPercentage`: 1-99, typically 10-25

**Limitation:** Uses tokenizer-based splitting but does NOT respect semantic boundaries (sentences, paragraphs). Splits may occur mid-sentence.

### 3.11.3 Strategy 3: Hierarchical Chunking (Default)

```python
kb_config = {
    "chunkingConfiguration": {
        "chunkingStrategy": "HIERARCHICAL",
        "hierarchicalChunkingConfiguration": {
            "levelConfigurations": [
                {
                    "maxTokens": 500,      # Small chunks
                    "overlapPercentage": 10,
                },
                {
                    "maxTokens": 2000,     # Larger parent chunks
                    "overlapPercentage": 5,
                },
            ],
            "overlapTokens": 50,  # Tokens of overlap between levels
        }
    }
}
```

**How it works:**
1. Creates small "child" chunks (500 tokens) for granular retrieval
2. Creates larger "parent" chunks (2000 tokens) for context
3. At query time, retrieves matching child chunks, then returns the parent document

**When to use:** Default recommended strategy. Good for most document types.

### 3.11.4 Strategy 4: Semantic Chunking

```python
kb_config = {
    "chunkingConfiguration": {
        "chunkingStrategy": "SEMANTIC",
        "semanticChunkingConfiguration": {
            "maxTokens": 500,
            "bufferSize": 1,           # Sentences to include before/after
            "breakpointPercentileThreshold": 95,  # Similarity threshold for splits
        }
    }
}
```

**How it works:**
1. Splits text into sentences
2. Groups sentences into paragraphs using embedding similarity
3. Splits at points where semantic similarity drops below threshold
4. `breakpointPercentileThreshold`: Higher = bigger chunks, Lower = smaller chunks

**When to use:** When document structure is important (essays, articles, reports). More compute-intensive than other strategies.

### 3.11.5 Custom Chunking (Pre-processing)

For full control, process documents before uploading to Bedrock KB:

```python
def custom_chunk_and_upload(documents: list[Document], s3_uri: str):
    """
    Pre-chunk documents with custom logic, then upload and configure KB
    to use pre-chunked content.
    """
    # Step 1: Custom chunk with your logic
    all_chunks = []
    for doc in documents:
        chunks = your_custom_chunker(doc)
        for i, chunk in enumerate(chunks):
            # Add custom metadata
            chunk.metadata = {
                "chunk_id": f"{doc.id}_{i}",
                "parent_doc_id": doc.id,
                "chunk_index": i,
                "section": detect_section(chunk),
            }
            all_chunks.append(chunk)
    
    # Step 2: Generate custom embeddings
    embeddings = titan_model.embed_documents([c.content for c in all_chunks])
    
    # Step 3: Write chunks + embeddings to S3 in Bedrock-compatible format
    # Bedrock expects JSONL with text and embeddings
    with open('/tmp/chunks.jsonl', 'w') as f:
        for chunk, embedding in zip(all_chunks, embeddings):
            f.write(json.dumps({
                "text": chunk.content,
                "embedding": embedding,
                "metadata": chunk.metadata,
            }) + '\n')
    
    # Step 4: Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file('/tmp/chunks.jsonl', bucket, key)
    
    # Step 5: Create KB with custom parser
    bedrock.create_knowledge_base(
        name="custom-chunked-kb",
        dataSourceConfiguration={
            "s3Configuration": {
                "bucketArn": bucket_arn,
                "inclusionPrefixes": [key],
            },
            "type": "CUSTOM",
        },
        vectorStoreConfiguration={
            "serviceConfiguration": {
                "type": "OPEN_SEARCH_SERVERLESS",
            }
        }
    )
```

## 3.12 Code-Specific Chunking Strategies

Code has natural boundaries at multiple levels:

```
Repository
  └── Directory
        └── File
              └── Class / Module
                    └── Function / Method
                          └── Code Block (control flow)
```

**Best practice:** Index at multiple granularities:

1. **File-level chunks** for "What does this file do?" queries
2. **Function-level chunks** for "Find me the authentication logic" queries
3. **Class-level chunks** for "What classes handle payments?" queries

```python
import ast

def chunk_python_code(source_code: str, file_path: str) -> list[dict]:
    """
    Parse Python code and create chunks at multiple granularities.
    """
    tree = ast.parse(source_code)
    
    chunks = []
    
    # File-level chunk (entire file)
    chunks.append({
        "content": source_code,
        "type": "file",
        "name": file_path,
        "size": len(source_code.split()),
    })
    
    # Class-level chunks
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_source = ast.get_source_segment(source_code, node)
            chunks.append({
                "content": class_source,
                "type": "class",
                "name": f"{file_path}::{node.name}",
                "parent": file_path,
                "size": len(class_source.split()),
            })
            
            # Function-level within class
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    func_source = ast.get_source_segment(source_code, item)
                    chunks.append({
                        "content": func_source,
                        "type": "function",
                        "name": f"{file_path}::{node.name}::{item.name}",
                        "parent": f"{file_path}::{node.name}",
                        "size": len(func_source.split()),
                    })
    
    return chunks
```

## 3.13 Chunk Quality Metrics

Measuring chunk quality helps you iterate on chunking strategies:

### 3.13.1 Coherence Score

```python
def coherence_score(chunk: str, embedding_model) -> float:
    """
    Measure how coherent a chunk is by computing embedding similarity
    between sentences within the chunk.
    
    High coherence = sentences are related to each other.
    """
    sentences = split_into_sentences(chunk)
    if len(sentences) < 2:
        return 1.0  # Single sentence, trivially coherent
    
    embeddings = embedding_model.embed_documents(sentences)
    
    # Average pairwise similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)
    
    return np.mean(similarities)
```

**Interpretation:** Coherence > 0.7 = good, < 0.4 = chunk likely has topic shifts

### 3.13.2 Completeness Score

```python
def completeness_score(chunk: str) -> float:
    """
    Measure whether a chunk is self-contained (has enough context to understand).
    
    Checks:
    - Does it start mid-sentence? (indicator: no capital letter, starts with article)
    - Does it end mid-thought? (indicator: no ending punctuation)
    - Is it too short? (< 3 sentences)
    """
    sentences = split_into_sentences(chunk)
    score = 1.0
    
    # Penalty for starting mid-sentence
    first_sentence = sentences[0].strip()
    if first_sentence and first_sentence[0].islower():
        score *= 0.8
    if first_sentence.startswith(('the ', 'a ', 'an ', 'this ', 'that ')):
        score *= 0.9  # Likely continuation
    
    # Penalty for ending mid-sentence
    last_sentence = sentences[-1].strip()
    if not last_sentence.endswith(('.', '!', '?', '"', "'")):
        score *= 0.8
    
    # Penalty for shortness
    if len(sentences) < 3:
        score *= 0.9
    
    return score
```

---

---

# Part 4: Vector Databases — Comprehensive Guide

## 4.1 What Makes a Database a "Vector Database"

A vector database is a specialized database system designed to store, index, and query high-dimensional vector embeddings. While traditional databases store exact matches (e.g., "find row where id = 42"), vector databases enable **similarity search** ("find the 10 most similar items to vector X").

Core capabilities that distinguish vector databases from regular databases with vector support:

| Capability | Traditional DB | Vector DB |
|-----------|---------------|-----------|
| Primary query | Exact match | Similarity search |
| Index type | B-tree, Hash | HNSW, IVF, etc. |
| Distance metrics | =, !=, <, > | Cosine, L2, Dot product |
| Scalability | Handles billions of rows | Optimized for billion-scale vectors |
| Filtering | SQL WHERE clauses | Pre-filter + post-filter hybrid |

## 4.2 ANN Algorithms: The Engineering Behind Vector Search

Exact nearest neighbor search has O(n) complexity — you must compare your query vector against every stored vector. For 100 million vectors, this is impractical. **Approximate Nearest Neighbor (ANN)** algorithms trade a small amount of accuracy for orders-of-magnitude speed improvement.

### 4.2.1 HNSW (Hierarchical Navigable Small World)

**HNSW** is the most widely used ANN algorithm. It builds a multi-layer graph where top layers are sparse (fast long-range navigation) and bottom layers are dense (precise local search). Search starts at the top, greedily traverses to the closest neighbor, then descends.

**Key parameters:**
- `M` (max connections per node): Higher = better recall, more memory, slower build. Typical: 16-64
- `efConstruction`: Size of dynamic candidate list during build. Higher = better quality, slower build. Typical: 100-400
- `efSearch`: Size of dynamic candidate list during search. Higher = better recall, slower search. Typical: 50-400

### 4.2.2 IVF (Inverted File Index)

IVF partitions the vector space into `k` clusters using k-means and builds an inverted index. For each cluster center, it stores the list of vectors belonging to that cluster. Search finds the closest cluster center and searches only that cluster's inverted list.

**Key parameters:** `nlist` (number of clusters: 1000-10000) and `nprobe` (clusters to search).

### 4.2.3 Product Quantization (PQ)

PQ compresses vectors by splitting them into sub-vectors and quantizing each independently. Enables 10-50x compression with acceptable recall loss (~85-95%).

### 4.2.4 Algorithm Comparison

| Algorithm | Recall | Query Latency | Index Size | Best For |
|-----------|--------|---------------|-----------|----------|
| HNSW | 95-99% | 1-10ms | 1.2-1.5x data | General purpose, high recall |
| IVF+PQ | 85-95% | 5-20ms | 0.1-0.2x data | Memory-constrained |
| IVF (flat) | 90-97% | 2-15ms | 1.0x data | Balance recall/memory |
| LSH | 70-85% | 1-5ms | 1.5-2x data | Ultra-low latency |

## 4.3 Exact vs. Approximate Search Trade-offs

Exact (brute force) search achieves 100% recall but requires comparing against every vector. For 1 million vectors at 1ms per comparison, that's 1 second per query. ANN algorithms like HNSW achieve 97-99% recall with only 30-50 distance computations, reducing query time to 5-10ms.

Use exact search when: dataset < 10,000 vectors, 100% recall is mandatory, or latency budget is >100ms.

Use ANN when: dataset > 100K vectors, latency < 50ms required, or cost constraints exist.

## 4.4 Index Types and Construction

### OpenSearch Serverless Index Configuration

OpenSearch Serverless uses faiss HNSW indexes by default. Key index settings:

```json
{
  "settings": {
    "index.knn": true,
    "index.knn.algo_param.ef_search": 100,
    "index.knn.algo_param.ef_construction": 200
  },
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "engine": "faiss",
          "parameters": {"m": 16, "ef_construction": 200}
        }
      }
    }
  }
}
```

## 4.5 Filtering and Hybrid Search

### 4.5.1 Pre-filter vs. Post-filter

**Pre-filtering:** Apply metadata filters BEFORE vector search, then search within the filtered subset. Problem: filtered subset may be small, making ANN less effective.

**Post-filtering:** Retrieve top-k results using vector similarity, then REMOVE results that don't match the filter. Problem: may end up with fewer than k results.

### 4.5.2 Hybrid Search: Dense + Sparse

Combining vector (dense) search with BM25/keyword (sparse) search often outperforms either alone using Reciprocal Rank Fusion (RRF):

```python
def hybrid_search_rrf(
    dense_results: list, sparse_results: list,
    k: int = 60, dense_weight: float = 0.7, sparse_weight: float = 0.3
) -> list[dict]:
    """Combine dense and sparse search with RRF fusion."""
    fused_scores = {}
    for rank, result in enumerate(dense_results):
        doc_id = result["doc_id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + dense_weight * (1 / (k + rank + 1))
    for rank, result in enumerate(sparse_results):
        doc_id = result["doc_id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + sparse_weight * (1 / (k + rank + 1))
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

## 4.6 Vector Database Comparison Table

| Database | Type | Max Dim | ANN Algorithm | Cloud Native | Best For |
|----------|------|---------|---------------|-------------|----------|
| **Pinecone** | SaaS | 100K | Proprietary HNSW | Yes | Fully managed |
| **Weaviate** | Self/SaaS | 65K | HNSW + hybrid | Partial | Graph + Vector |
| **Chroma** | In-process | 4K | HNSW | No | Local dev |
| **Milvus** | Self/SaaS | 32K | HNSW, IVF, PQ | Partial | Scale |
| **Qdrant** | Self/SaaS | 65K | HNSW | Partial | Filtering |
| **pgvector** | PostgreSQL ext | 16K | HNSW, IVFFlat | Yes | Existing infra |
| **OpenSearch Serverless** | AWS SaaS | 16K | HNSW (faiss) | Yes (AWS) | AWS-native |
| **Aurora pgvector** | AWS RDS | 16K | HNSW, IVFFlat | Yes (AWS) | Aurora users |
| **FAISS** | Library | Unlimited | Many | No | Local processing |

## 4.7 Amazon OpenSearch Serverless Vector Engine — Architecture

OpenSearch Serverless is the default vector store for Bedrock Knowledge Bases. It auto-scales compute based on query volume with no server management.

### Cost Model (us-east-1):
- **Storage**: $0.24 per GB per month
- **Compute**: $0.024 per OCU-hour (1 OCU = ~2 vCPU + 8 GB RAM)
- **Example**: 1M chunks (1536-dim) costs ~$52-100/month at moderate query volume

### Limits:
| Resource | Limit |
|----------|-------|
| Max dimensions | 16,384 |
| Max vectors per collection | 100M (soft) |
| Max collections per account | 50 |
| Query timeout | 30 seconds |
| Max k for k-NN | 10,000 |

## 4.8 Aurora PostgreSQL with pgvector

pgvector enables vector similarity search within Aurora PostgreSQL. Key advantages: ACID transactions, read replicas for scaling, existing Aurora infrastructure.

```python
# Aurora pgvector schema
CREATE TABLE IF NOT EXISTS bedrock_docs (
    id UUID PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    document_type VARCHAR(100)
);

-- HNSW index for high recall
CREATE INDEX idx_hnsw ON bedrock_docs USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Query with filtering
SELECT id, chunk_id, content, 1 - (embedding <=> %s::vector) AS similarity
FROM bedrock_docs WHERE document_type = %s
ORDER BY embedding <=> %s::vector LIMIT %s;
```

## 4.9 Memory-Based Stores: FAISS and Chroma

### FAISS

Facebook's library for efficient dense vector similarity search. Designed for in-memory scenarios.

```python
import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension: int = 1536, m: int = 16):
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(dimension, m)
        self.index.hnsw.efSearch = 100
        self.index.hnsw.efConstruction = 200
        self.docstore = {}
        self._next_id = 0
    
    def add(self, vectors: list[list[float]], documents: list[dict]):
        vectors_np = np.array(vectors).astype('float32')
        start_id = self._next_id
        self.index.add(vectors_np)
        for i, doc in enumerate(documents):
            self.docstore[start_id + i] = doc
        self._next_id += len(vectors)
    
    def search(self, query_vector: list[float], k: int = 10) -> list[dict]:
        query_np = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_np, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self.docstore:
                result = self.docstore[idx].copy()
                result["distance"] = float(dist)
                result["similarity"] = 1.0 / (1.0 + result["distance"])
                results.append(result)
        return results
    
    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docstore.json", "w") as f:
            json.dump(self.docstore, f)
```

### ChromaDB

Open-source embedded vector database optimized for simplicity in development.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# Add documents (auto-embeds if embedding function configured)
collection.add(
    documents=["doc1 text", "doc2 text"],
    ids=["id1", "id2"],
    metadatas=[{"source": "manual"}, {"source": "auto"}],
)

# Query
results = collection.query(
    query_texts=["query text"],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
```

**Chroma limitations:** Single-node only, max dataset size limited by disk/memory, not production-grade for large scale.

---

# Part 5: The RAG Pipeline — Step by Step

## 5.1 High-Level Architecture

A RAG system has two distinct pipelines: **ingestion** (building the knowledge base) and **query** (answering questions).

```
INGESTION PIPELINE (offline):
  Documents → Parse & Extract → Chunk → Embed → Index → Store

QUERY PIPELINE (online):
  Query → Embed → Retrieve → Re-rank → Assemble Context → Generate → Response + Citations
```

## 5.2 Ingestion Pipeline

### 5.2.1 Document Loading

```python
from langchain.document_loaders import (
    PyPDFLoader, UnstructuredHTMLLoader, ConfluenceLoader,
    S3FileLoader, UnstructuredMarkdownLoader, TextLoader,
)

def load_documents(config: list[dict]) -> list[Document]:
    """Load documents from various sources."""
    loaders = {
        "pdf": PyPDFLoader,
        "html": UnstructuredHTMLLoader,
        "confluence": ConfluenceLoader,
        "s3": S3FileLoader,
        "markdown": UnstructuredMarkdownLoader,
    }
    all_docs = []
    for item in config:
        loader_type = item["source_type"]
        loader = loaders[loader_type](**item["params"])
        all_docs.extend(loader.load())
    return all_docs
```

### 5.2.2 Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=lambda x: len(x.split()),
)

def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
```

### 5.2.3 Embedding

```python
import boto3
import json

class BedrockEmbedder:
    def __init__(self, model_id: str = "amazon.titan-embed-text-v2", region: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.batch_size = 100
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings
    
    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"inputText": texts if len(texts) > 1 else texts[0]})
        response = self.client.invoke_model(
            modelId=self.model_id, body=body,
            accept="application/json", contentType="application/json",
        )
        response_body = json.loads(response["body"].read().decode())
        return response_body.get("embeddings", [response_body["embedding"]])
```

## 5.3 Query Pipeline

```python
class RAGQueryPipeline:
    def __init__(self, embedder, vector_store, llm, reranker=None, k_retrieve=20, k_final=5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker
        self.k_retrieve = k_retrieve
        self.k_final = k_final
    
    def query(self, question: str, metadata_filter: dict = None) -> dict:
        # 1. Embed query
        query_vector = self.embedder.embed_query(question)
        
        # 2. Retrieve candidates
        candidates = self.vector_store.similarity_search(
            query_vector=query_vector, k=self.k_retrieve, filter=metadata_filter,
        )
        
        # 3. Re-rank
        if self.reranker:
            ranked = self.reranker.rerank(question, candidates, self.k_final)
        else:
            ranked = candidates[:self.k_final]
        
        # 4. Assemble context
        context = self._assemble_context(ranked)
        
        # 5. Generate answer
        answer = self._generate_answer(question, context)
        
        # 6. Extract citations
        citations = self._extract_citations(ranked)
        
        return {"answer": answer, "citations": citations, "source_chunks": ranked}
    
    def _assemble_context(self, chunks: list) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(f"[Source {i+1}]\n{chunk.page_content}\n")
        return "\n---\n".join(parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful assistant. Use the provided context to answer.
If the answer is not in the context, say you don't know. Always cite your sources.

Context:
{context}

Question: {question}

Answer:"""
        return self.llm.invoke(prompt)
```

## 5.4 Synchronous vs. Asynchronous Ingestion

**Synchronous:** Steps run sequentially. Simple but slow. Use for small datasets or debugging.

**Asynchronous:** Use parallel processing for independent steps (parsing, embedding batches).

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_ingest(documents: list, max_workers: int = 10):
    """Asynchronous ingestion with parallel processing."""
    loop = asyncio.get_event_loop()
    
    # Parse documents in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        parsed = await asyncio.gather(*[
            loop.run_in_executor(executor, preprocess, doc)
            for doc in documents
        ])
    
    # Chunk all parsed documents
    all_chunks = []
    for doc in parsed:
        all_chunks.extend(chunk(doc))
    
    # Embed in batches (parallel batches)
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = [c.page_content for c in all_chunks[i:i+batch_size]]
        emb = await loop.run_in_executor(executor, embedder.embed_documents, batch)
        all_embeddings.extend(emb)
    
    # Index all
    for chunk, emb in zip(all_chunks, all_embeddings):
        index(chunk, emb)
    
    return {"chunks_created": len(all_chunks), "status": "complete"}
```

## 5.5 Incremental Updates

```python
class IncrementalIndexer:
    def check_for_updates(self, documents: list) -> dict:
        """Detect new, modified, unchanged, and deleted documents."""
        changes = {}
        for doc in documents:
            doc_id = doc.metadata.get("doc_id")
            existing = self.docstore.get(doc_id)
            if existing is None:
                changes[doc_id] = "new"
            elif existing.page_content != doc.page_content:
                changes[doc_id] = "modified"
            else:
                changes[doc_id] = "unchanged"
        return changes
    
    def apply_updates(self, documents: list, changes: dict) -> dict:
        """Apply incremental updates based on change detection."""
        stats = {"added": 0, "updated": 0, "deleted": 0}
        for doc in documents:
            doc_id = doc.metadata.get("doc_id")
            change = changes.get(doc_id)
            if change == "new":
                self._index_document(doc)
                stats["added"] += 1
            elif change == "modified":
                self._delete_document(doc_id)
                self._index_document(doc)
                stats["updated"] += 1
        return stats
```

---

---

# Part 6: Retrieval Strategies — Deep Dive

Retrieval is the most impactful stage of a RAG pipeline. Poor retrieval cannot be fixed by a better LLM. This section covers every major retrieval strategy from basic vector search to advanced re-ranking.

## 6.1 Dense Retrieval (Vector Similarity)

Dense retrieval uses neural network embeddings to capture semantic meaning. It's the foundation of modern RAG.

### 6.1.1 How It Works

1. At indexing time: each document chunk is embedded into a dense vector
2. At query time: the query is embedded using the same model
3. Similarity search finds the nearest document vectors to the query vector

```python
def dense_retrieve(
    query_vector: list[float],
    k: int = 10,
    filter_metadata: dict = None,
) -> list[dict]:
    """
    Retrieve documents using vector similarity.
    
    Most vector databases use approximate nearest neighbor (ANN)
    algorithms like HNSW for efficiency at scale.
    """
    results = vector_index.search(
        query_vector=query_vector,
        k=k,
        method="hnsw",  # Hierarchical Navigable Small World
        filters=filter_metadata,
    )
    return results
```

### 6.1.2 Strengths and Limitations

**Strengths:**
- Captures semantic similarity (synonyms, paraphrases)
- "Knows" that "puppy" and "dog" are related even without shared terms
- Works across languages with multilingual embeddings

**Limitations:**
- Requires good embedding model quality
- May miss exact keyword matches
- Sensitive to embedding model domain mismatch (medical embeddings on legal docs)

## 6.2 Sparse Retrieval (BM25, TF-IDF)

Sparse retrieval uses traditional information retrieval: term frequency and inverse document frequency. Results are sparse vectors where non-zero dimensions correspond to specific terms.

### 6.2.1 BM25 Algorithm

BM25 (Best Matching 25) is the de facto standard for sparse retrieval:

```
BM25Score(d, q) = Σ IDF(qi) × (tf(ti, d) × (k1 + 1)) / (tf(ti, d) + k1 × (1 - b + b × |d|/avgdl))

Where:
- tf(ti, d) = term frequency of term ti in document d
- |d| = document length
- avgdl = average document length in corpus
- k1 = term saturation parameter (typically 1.2-2.0)
- b = length normalization parameter (typically 0.75)
- IDF(qi) = inverse document frequency of term qi
```

**Key insight:** BM25 uses term saturation — repeating a term more times has diminishing returns. A term appearing 10 times doesn't score 10x higher than appearing once.

### 6.2.2 Implementation

```python
from rank_bm25 import BM25Okapi
import re

class BM25Retriever:
    def __init__(self, corpus: list[str]):
        """Build BM25 index from corpus."""
        # Tokenize corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.corpus = corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def retrieve(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """
        Retrieve top-k documents for query.
        Returns list of (doc_index, score) tuples sorted by score.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k document indices
        top_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return top_indices
    
    def get_doc(self, index: int) -> str:
        return self.corpus[index]
```

## 6.3 Hybrid Retrieval: Combining Dense + Sparse

Hybrid retrieval combines the semantic power of dense search with the precision of sparse keyword search.

### 6.3.1 Reciprocal Rank Fusion (RRF)

RRF is the most common fusion method. It combines rankings from multiple retrievers:

```python
def reciprocal_rank_fusion(
    result_lists: list[list[tuple[doc_id, score]]],
    k: int = 60,
    weights: list[float] = None,
) -> list[tuple[doc_id, float]]:
    """
    Combine multiple retrieval result lists using RRF.
    
    RRF Score = Σ weight_i × (1 / (k + rank_i))
    
    Args:
        result_lists: List of retrieval results, each is list of (doc_id, score)
        k: RRF parameter (default 60, higher = more uniform distribution)
        weights: Weight for each result list (default: equal weights)
    
    Returns:
        Sorted list of (doc_id, fused_score)
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    fused = {}
    
    for result_list, weight in zip(result_lists, weights):
        for rank, (doc_id, _) in enumerate(result_list):
            # RRF formula with weight
            rrf_score = weight * (1 / (k + rank + 1))
            fused[doc_id] = fused.get(doc_id, 0) + rrf_score
    
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

### 6.3.2 Complete Hybrid Retriever

```python
class HybridRetriever:
    def __init__(self, dense_index, sparse_corpus):
        self.dense_index = dense_index
        self.bm25 = BM25Retriever(sparse_corpus)
    
    def retrieve(
        self,
        query: str,
        query_vector: list[float],
        k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict]:
        # Dense retrieval
        dense_results = self.dense_index.search(query_vector, k=k*2)
        
        # Sparse retrieval
        sparse_results = self.bm25.retrieve(query, k=k*2)
        
        # RRF fusion
        fused = reciprocal_rank_fusion(
            result_lists=[
                [(r["doc_id"], r["score"]) for r in dense_results],
                [(idx, score) for idx, score in sparse_results],
            ],
            k=60,
            weights=[dense_weight, sparse_weight],
        )
        
        # Return top k with scores
        return [{"doc_id": doc_id, "fused_score": score} for doc_id, score in fused[:k]]
```

## 6.4 Query Expansion Techniques

Query expansion improves recall by reformulating the user's query to better match document language.

### 6.4.1 Multi-Query Expansion

Generate multiple reformulated versions of the query and retrieve from all:

```python
def multi_query_expansion(
    query: str,
    llm,
    num_variants: int = 5,
) -> list[str]:
    """
    Generate multiple query variants using an LLM.
    """
    prompt = f"""Generate {num_variants} different ways to express this query.
    Create variations that:
    1. Use synonyms for key terms
    2. Rephrase the question structure
    3. Narrow or broaden the scope
    4. Use different but related terminology
    
    Return only the variations, one per line, no numbering.
    
    Original query: {query}"""
    
    response = llm.invoke(prompt)
    variants = [query] + [v.strip() for v in response.split('\n') if v.strip()]
    return variants[:num_variants]

def expanded_retrieve(
    query: str,
    embedder,
    retriever,
    llm,
    k_per_variant: int = 5,
) -> list[dict]:
    """
    Retrieve using multiple query variants and fuse results.
    """
    variants = multi_query_expansion(query, llm)
    
    all_results = []
    for variant in variants:
        vector = embedder.embed_query(variant)
        results = retriever.search(query_vector=vector, k=k_per_variant)
        all_results.append([(r["doc_id"], r["score"]) for r in results])
    
    # Fused results across all variants
    fused = reciprocal_rank_fusion(all_results, k=60)
    return fused
```

### 6.4.2 HyDE (Hypothetical Document Embeddings)

HyDE generates a hypothetical "ideal" document and uses its embedding for retrieval:

```python
defhyde_retrieve(query: str, embedder, llm, retriever) -> list[dict]:
    """
    HyDE: Generate a hypothetical document answering the query,
    then embed and retrieve using that instead of the query.
    
    Intuition: The hypothetical document shares the embedding space
    of real documents, so it retrieves more similar documents.
    """
    # Generate hypothetical document
    prompt = f"""Write a short passage (2-3 sentences) that directly answers this question.
    The passage should be factual and detailed.
    
    Question: {query}
    
    Hypothetical passage:"""
    
    hypothetical_doc = llm.invoke(prompt)
    
    # Embed the hypothetical document (NOT the query)
    doc_vector = embedder.embed_query(hypothetical_doc)
    
    # Retrieve using the hypothetical document's embedding
    results = retriever.search(query_vector=doc_vector, k=10)
    
    return results, hypothetical_doc
```

## 6.5 Query Rewriting

Query rewriting modifies the query to improve retrieval, especially for conversational RAG where context matters.

### 6.5.1 Step-Back Prompting

```python
def step_back_query(query: str, llm) -> str:
    """
    Step-back prompting: abstract the query to a higher-level concept.
    Then retrieve using both the original and the step-back query.
    """
    prompt = f"""Given this query, formulate a more abstract, higher-level question
    that captures the core concept being asked about.
    
    Query: {query}
    
    Step-back question:"""
    
    step_back = llm.invoke(prompt)
    return step_back

def dual_query_retrieve(query: str, embedder, retriever, llm) -> list[dict]:
    # Get step-back query
    step_back = step_back_query(query, llm)
    
    # Embed both queries
    query_vec = embedder.embed_query(query)
    step_back_vec = embedder.embed_query(step_back)
    
    # Retrieve from both
    results = retriever.search(query_vector=query_vec, k=10)
    step_back_results = retriever.search(query_vector=step_back_vec, k=10)
    
    # Fuse results
    fused = reciprocal_rank_fusion([
        [(r["doc_id"], r["score"]) for r in results],
        [(r["doc_id"], r["score"]) for r in step_back_results],
    ])
    
    return fused
```

## 6.6 Contextual Compression

After retrieval, compress each document to only the most relevant passages:

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

class ContextualCompressor:
    def __init__(self, llm):
        self.llm = llm
    
    def compress(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Extract only the sentences relevant to the query from each document.
        Uses an LLM to identify relevant spans.
        """
        compressed = []
        
        for doc in documents:
            prompt = f"""Extract only the sentences from this document that are relevant
            for answering the query. Return the exact sentences, no rephrasing.
            
            Query: {query}
            
            Document:
            {doc.page_content}
            
            Relevant sentences:"""
            
            relevant_text = self.llm.invoke(prompt)
            
            if relevant_text.strip():
                compressed_doc = Document(
                    page_content=relevant_text,
                    metadata=doc.metadata,
                )
                compressed.append(compressed_doc)
        
        return compressed
```

## 6.7 Sub-Document Retrieval (Parent Document Retriever)

Retrieve at chunk level but return parent documents for richer context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

class ParentDocumentArchitecture:
    """
    Two-level retrieval:
    1. Small chunks for precise retrieval (500 tokens)
    2. Parent documents for full context (2000 tokens)
    """
    
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.docstore = InMemoryStore()  # Stores parent documents
    
    def setup(self, documents: list[Document]):
        # Parent splitter (large chunks)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200,
        )
        
        # Child splitter (small chunks)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
        )
        
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
        )
        
        self.retriever.add_documents(documents)
    
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """
        Retrieve matching child chunks, then return their parent documents.
        """
        # This retrieves children, then fetches parents
        return self.retriever.get_relevant_documents(query, k=k)
```

## 6.8 Re-ranking: Cross-Encoders and Learning-to-Rank

After initial retrieval (designed for recall), re-ranking optimizes for precision by scoring the top candidates with a more expensive but more accurate model.

### 6.8.1 Cross-Encoder Re-ranking

Cross-encoders jointly encode query and document (unlike bi-encoders which encode separately). This is more accurate but too slow for initial retrieval over millions of documents.

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank documents using a cross-encoder.
        Cross-encoder jointly encodes [query, document] pair.
        """
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Score all pairs (slower but more accurate)
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [
            {"document": doc, "score": float(score), "rank": i + 1}
            for i, (doc, score) in enumerate(scored_docs[:top_k])
        ]

# Usage in pipeline
reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Initial retrieval (fast, recall-oriented)
candidates = vector_store.search(query_vector=query_emb, k=50)

# Re-ranking (slower, precision-oriented)
reranked = reranker.rerank(query, [c["content"] for c in candidates], top_k=5)
```

### 6.8.2 Cohere Rerank (API-based)

```python
import cohere

cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

def cohere_rerank(
    query: str,
    documents: list[str],
    top_n: int = 5,
    model: str = "rerank-multilingual-v3.0",
) -> list[dict]:
    """
    Use Cohere's API for production-grade reranking.
    Supports multilingual documents.
    """
    response = cohere_client.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model=model,
        return_documents=True,
    )
    
    return [
        {
            "index": result.index,
            "document": result.document.text,
            "score": result.relevance_score,
        }
        for result in response.results
    ]
```

## 6.9 Maximum Marginal Relevance (MMR)

MMR balances relevance with diversity in retrieved results. Without it, top results tend to be redundant (same source, same viewpoint).

```python
def maximum_marginal_relevance(
    query_vector: list[float],
    doc_vectors: list[list[float]],
    doc_contents: list[str],
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> list[dict]:
    """
    MMR: Balance relevance (semantic similarity) with diversity.
    
    MMR Score = λ × relevance(query, doc) - (1 - λ) × max_similarity(doc, selected)
    
    lambda_mult:
    - 1.0 = maximum relevance (like standard retrieval)
    - 0.0 = maximum diversity
    - 0.5 = balanced (default)
    
    Args:
        query_vector: Embedded query
        doc_vectors: Embedded documents to select from
        doc_contents: Original document text
        k: Number of results to return
        fetch_k: Number of candidates to consider
        lambda_mult: Balance parameter
    """
    selected = []
    selected_indices = []
    remaining_indices = list(range(len(doc_vectors)))
    
    # Get top fetch_k by relevance
    initial_scores = [cosine_similarity(query_vector, v) for v in doc_vectors]
    scored = sorted(enumerate(initial_scores), key=lambda x: x[1], reverse=True)
    remaining_indices = [idx for idx, _ in scored[:fetch_k]]
    
    while len(selected) < k and remaining_indices:
        best_score = float('-inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Relevance to query
            relevance = cosine_similarity(query_vector, doc_vectors[idx])
            
            # Diversity: max similarity to already selected
            if selected_indices:
                similarities = [cosine_similarity(doc_vectors[idx], doc_vectors[s])
                              for s in selected_indices]
                max_sim = max(similarities)
            else:
                max_sim = 0
            
            # MMR score
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append({
                "index": best_idx,
                "content": doc_contents[best_idx],
                "mmr_score": best_score,
            })
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return selected
```

## 6.10 Metadata Filtering Strategies

Metadata filtering narrows retrieval to relevant document subsets before or after vector search.

```python
class MetadataFilteredRetriever:
    def retrieve(
        self,
        query_vector: list[float],
        filters: dict,
        k: int = 10,
    ) -> list[dict]:
        """
        Retrieve with metadata filtering.
        
        Supported filter operators:
        - Exact match: {"field": "value"}
        - Range: {"field": {"$gte": value}}
        - IN: {"field": {"$in": [val1, val2]}}
        - AND: {"$and": [filter1, filter2]}
        - OR: {"$or": [filter1, filter2]}
        """
        filter_clause = self._build_filter_clause(filters)
        
        return self.vector_store.similarity_search(
            query_vector=query_vector,
            k=k,
            filter=filter_clause,
        )
    
    def _build_filter_clause(self, filters: dict) -> dict:
        """Convert Python dict to OpenSearch filter syntax."""
        clauses = []
        
        for field, value in filters.items():
            if isinstance(value, dict):
                # Range query
                for op, val in value.items():
                    if op == "$gte":
                        clauses.append({"range": {field: {"gte": val}}})
                    elif op == "$lte":
                        clauses.append({"range": {field: {"lte": val}}})
                    elif op == "$in":
                        clauses.append({"terms": {field: val}})
            else:
                # Exact match
                clauses.append({"term": {field: value}})
        
        if len(clauses) == 1:
            return clauses[0]
        elif len(clauses) > 1:
            return {"bool": {"filter": clauses}}
        else:
            return {}

# Example usage
results = retriever.retrieve(
    query_vector=query_emb,
    filters={
        "document_type": "policy",
        "version": {"$gte": "2024-01"},
        "department": {"$in": ["engineering", "product"]},
    },
    k=10,
)
```

---

# Part 7: Context Assembly & Prompt Engineering for RAG

## 7.1 How Many Chunks to Include (k Selection)

The optimal number of chunks depends on:
- **Chunk size**: Smaller chunks → more chunks needed for context
- **LLM context window**: Must fit prompt + chunks + answer
- **Query complexity**: Simple questions need less context
- **Retrieval quality**: If top chunks are very relevant, fewer needed

### 7.1.1 Practical Guidelines

| Chunk Size | Typical k | Max Tokens (8K window) |
|------------|-----------|----------------------|
| 200 tokens | 15-20 | ~3,000 + prompt + answer |
| 500 tokens | 8-10 | ~5,000 + prompt + answer |
| 1000 tokens | 4-5 | ~5,000 + prompt + answer |

### 7.1.2 Adaptive k Selection

```python
def adaptive_chunk_selection(
    question: str,
    candidates: list[Document],
    llm,
    min_chunks: int = 3,
    max_chunks: int = 10,
    target_tokens: int = 4000,
) -> list[Document]:
    """
    Adaptively select number of chunks based on:
    - Question complexity (estimated via LLM)
    - Available token budget
    - Candidate relevance scores
    """
    # Estimate question complexity
    complexity_prompt = f"""Rate the complexity of this question on a scale of 1-10.
    1 = simple factual question, 10 = complex multi-part analysis.
    Also estimate how much context (number of paragraphs) would be needed.
    
    Question: {question}
    
    Respond in format: complexity=X, context_needed=Y"""
    
    response = llm.invoke(complexity_prompt)
    # Parse complexity and context_needed from response
    complexity = int(parse(response, "complexity=(\d)"))
    context_needed = int(parse(response, "context_needed=(\d)"))
    
    # Adjust k based on complexity
    if complexity <= 3:
        k = min(min_chunks + 1, max_chunks)
    elif complexity <= 6:
        k = min(min_chunks + 3, max_chunks)
    else:
        k = min(context_needed + 2, max_chunks)
    
    # Also respect token budget
    total_tokens = 0
    selected = []
    for chunk in candidates:
        chunk_tokens = len(chunk.page_content.split()) * 1.3  # Approximate
        if total_tokens + chunk_tokens <= target_tokens and len(selected) < k:
            selected.append(chunk)
            total_tokens += chunk_tokens
    
    return selected
```

## 7.2 Ordering Chunks in Context

Order matters. The LLM tends to weight earlier and later context more heavily (primacy and recency effects).

**Recommended order:**
1. Most relevant chunks first (highest similarity score)
2. Then supporting/contextual chunks
3. Keep related chunks together

```python
def assemble_context(
    chunks: list[Document],
    question: str,
    scores: list[float] = None,
) -> str:
    """
    Assemble chunks into context string with source markers.
    
    Order by: relevance score descending, then document order.
    """
    # Sort by score descending
    if scores:
        sorted_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        chunks, scores = zip(*sorted_chunks)
        chunks, scores = list(chunks), list(scores)
    
    # Group chunks by source document (keep related content together)
    grouped = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(chunk)
    
    # Flatten with source markers
    parts = []
    for source, source_chunks in grouped.items():
        # Sort within source by chunk index
        source_chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        
        combined = "\n\n".join(c.page_content for c in source_chunks)
        parts.append(f"[From: {source}]\n{combined}")
    
    return "\n\n---\n\n".join(parts)
```

## 7.3 Handling Long Contexts: Lost in the Middle

Research shows LLMs struggle with information in the middle of long contexts. Techniques to mitigate:

### 7.3.1 Key-Value Caching of Relevant Spans

```python
def place_important_info_at_ends(context: str, important_spans: list[str]) -> str:
    """
    Position the most important retrieved content at the beginning
    and end of the context (primacy/recency effect).
    
    Strategy: Put highest-relevance content at both ends.
    """
    parts = context.split("\n\n---\n\n")
    
    if len(parts) <= 3:
        return context  # Not long enough to worry
    
    # Sort by relevance (first part is most relevant)
    # Move first and last to boundaries
    return parts[0] + "\n\n... [additional context] ...\n\n" + parts[-1]
```

### 7.3.2 Summarize Long Contexts

```python
def summarize_long_context(context: str, question: str, llm, max_tokens: int = 2000) -> str:
    """
    If context is very long, summarize it to fit within token budget.
    Keep information relevant to the question.
    """
    context_tokens = count_tokens(context)
    
    if context_tokens <= max_tokens:
        return context
    
    prompt = f"""Summarize the following context, keeping only information
    relevant to answering this question. Preserve key facts and citations.
    Keep the summary under {max_tokens} tokens.
    
    Question: {question}
    
    Context:
    {context}
    
    Summary:"""
    
    return llm.invoke(prompt)
```

## 7.4 System Prompts for RAG

```python
RAG_SYSTEM_PROMPT = """You are a helpful research assistant. Your task is to answer questions
based ONLY on the provided context. Follow these rules strictly:

1. ANSWER ONLY FROM CONTEXT: If the answer is not in the context, say exactly:
   "I don't have enough information in the provided context to answer this question."
   Do NOT guess or use external knowledge.

2. CITATION REQUIREMENT: For every factual claim, cite the source using [Source N] notation.
   If using information from Source 1, write [Source 1] after that fact.

3. PARTIAL ANSWERS: If the context contains SOME relevant information but not complete,
   answer with what is available and explicitly state what information is missing.

4. CONFLICTING INFORMATION: If sources conflict, acknowledge the conflict and present
   both perspectives, citing each source.

5. TONE: Be precise and concise. Avoid speculation. When uncertain, say so.

Context:
{context}

Question: {question}

Answer:"""

def build_rag_prompt(question: str, context: str, system_prompt: str = None) -> list[dict]:
    """Build a RAG prompt with proper message formatting."""
    if system_prompt is None:
        system_prompt = RAG_SYSTEM_PROMPT
    
    return [
        {"role": "system", "content": system_prompt.format(context=context, question=question)},
        {"role": "user", "content": question},
    ]
```

## 7.5 Citation and Attribution

```python
def extract_citations(retrieved_chunks: list[Document]) -> list[dict]:
    """
    Extract structured citations from retrieved chunks.
    """
    citations = []
    
    for i, chunk in enumerate(retrieved_chunks):
        metadata = chunk.metadata
        citation = {
            "source_num": i + 1,
            "source": metadata.get("source", "unknown"),
            "title": metadata.get("title", metadata.get("source", "Document")),
            "excerpt": chunk.page_content[:300] + "...",
            "url": metadata.get("url"),
            "page": metadata.get("page"),
            "relevance_score": metadata.get("score"),
        }
        citations.append(citation)
    
    return citations

def format_response_with_citations(answer: str, citations: list[dict]) -> str:
    """Format answer with inline source numbers and footnotes."""
    # Add inline citations to answer
    formatted = answer
    
    # Add citations section
    citations_section = "\n\n---\n## Sources\n"
    for cite in citations:
        citations_section += f"[{cite['source_num']}] {cite['title']}"
        if cite.get('page'):
            citations_section += f" (page {cite['page']})"
        if cite.get('url'):
            citations_section += f" - {cite['url']}"
        citations_section += "\n"
    
    return formatted + citations_section
```

## 7.6 Handling "No Relevant Documents Found"

```python
NO_DOCS_RESPONSE = """I searched my knowledge base but did not find documents
that directly answer your question. This could mean:

1. The information may be in a different document or section
2. The terminology used may differ from what's in my knowledge base
3. The information may not yet be in my knowledge base

I can help you by:
- Rewriting your question with different keywords
- Searching a broader set of documents
- Connecting you with a human expert for this topic

How would you like to proceed?"""

def handle_empty_retrieval(question: str, threshold: float = 0.5) -> str:
    """
    When retrieval returns no confident results, respond appropriately.
    """
    return NO_DOCS_RESPONSE

def check_retrieval_confidence(
    results: list[dict],
    threshold: float = 0.5,
) -> bool:
    """
    Check if top retrieval result meets minimum confidence threshold.
    """
    if not results:
        return False
    
    top_score = results[0].get("score", 0)
    return top_score >= threshold
```

---

---

# Part 8: Amazon Bedrock Knowledge Bases — Complete Deep Dive

This is the most important section for AWS practitioners. Bedrock Knowledge Bases is a **fully managed RAG service** that handles document ingestion, chunking, embedding, indexing, and retrieval with just a few API calls or console clicks. Understanding it deeply is essential for production AWS AI implementations.

## 8.1 What Are Bedrock Knowledge Bases?

Bedrock Knowledge Bases is an AWS managed service that provides:
- **Document ingestion** from various data sources (S3, Confluence, SharePoint, Salesforce, web crawlers)
- **Automatic chunking** with configurable strategies
- **Embedding generation** using Titan Embeddings or Cohere
- **Vector storage** using OpenSearch Serverless, Aurora pgvector, Pinecone, Redis, or MongoDB Atlas
- **Managed retrieval** via `RetrieveAndGenerate` and `Retrieve` APIs
- **Sync job management** for keeping knowledge bases up-to-date

### 8.1.1 Key Value Proposition

| Without Knowledge Bases | With Bedrock Knowledge Bases |
|------------------------|-----------------------------|
| Build custom ingestion pipeline | Managed ingestion |
| Manage embedding infrastructure | One-click embedding setup |
| Configure and scale vector DB | Fully managed vector store |
| Write retrieval + generation logic | Built-in RAG orchestration |
| Handle sync/updates manually | Managed sync jobs |

## 8.2 Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Amazon Bedrock Knowledge Bases Architecture               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA SOURCES                         INGESTION                      RETRIEVAL│
│  ───────────                         ────────                      ─────────│
│  ┌─────────────┐                    ┌───────────┐                 ┌────────┐│
│  │ Amazon S3   │──────────────►    │  Bedrock  │                │ Bedrock ││
│  │ (PDF, HTML, │                   │  Ingestion│                │  Model ││
│  │  JSON, TXT) │                   │   Service │                │  Router││
│  └─────────────┘                   └─────┬─────┘                └────┬───┘│
│  ┌─────────────┐                         │                          │    │
│  │ Confluence  │────┐                    ▼                          ▼    │
│  └─────────────┘    │           ┌───────────────┐           ┌──────────┐│
│  ┌─────────────┐    ├─────────►│ Chunk & Parse │           │ Retrieve ││
│  │ SharePoint  │    │           └───────┬───────┘           │    API   ││
│  └─────────────┘    │                   │                   └────┬─────┘│
│  ┌─────────────┐    │                   ▼                          │    │
│  │  Salesforce │    │           ┌───────────────┐                  │    │
│  └─────────────┘    │           │   Embedding   │                  │    │
│  ┌─────────────┐    │           │    Service    │                  │    │
│  │Web Crawler  │────┘           │(Titan/Cohere) │                  │    │
│  └─────────────┘                └───────┬───────┘                  │    │
│                                         │                          │    │
│                                         ▼                          │    │
│                                 ┌───────────────┐                  │    │
│                                 │  Vector Store │◄─────────────────┘    │
│                                 │ (OpenSearch,  │                         │
│                                 │Aurora, etc.)  │                         │
│                                 └───────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2.1 Data Flow

**Ingestion Flow:**
1. Data source connector pulls documents
2. Parser extracts text from documents (handling PDF, HTML, etc.)
3. Chunker splits documents using configured strategy
4. Embedding service generates vectors (Titan or Cohere)
5. Chunks + vectors stored in configured vector store
6. Sync status updated in Bedrock

**Query Flow:**
1. User query → Bedrock `RetrieveAndGenerate` API
2. Query embedded using same embedding model as ingestion
3. Vector search in configured vector store
4. Retrieved chunks assembled into prompt
5. LLM generates answer using prompt + retrieved context
6. Response returned with citations

## 8.3 Complete Setup Walkthrough

### 8.3.1 Console Setup

**Step 1: Create a Knowledge Base**

1. Navigate to Amazon Bedrock → Knowledge Bases → Create knowledge base
2. Enter name and optional description
3. Select IAM permissions (create new or use existing service role)
4. Choose embedding model (Titan Embeddings v2 recommended)
5. Select vector store (OpenSearch Serverless recommended)

**Step 2: Configure Data Source**

1. Select data source type (S3, Confluence, etc.)
2. Configure source-specific settings (bucket name, credentials, etc.)
3. Configure chunking strategy (see Section 8.6)
4. Set sync schedule (on-demand, scheduled, or event-driven)

**Step 3: Sync and Test**

1. Run initial sync job
2. Monitor sync status in Bedrock console
3. Test retrieval via console's "Test" button or API

### 8.3.2 API Setup with boto3

```python
import boto3
import json
import time

bedrock = boto3.client('bedrock', region_name='us-east-1')
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

# ─────────────────────────────────────────────────────────────
# STEP 1: Create IAM Role for Knowledge Base
# ─────────────────────────────────────────────────────────────

iam = boto3.client('iam')

trust_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}

assume_role_response = iam.create_role(
    RoleName="bedrock-kb-access-role",
    AssumeRolePolicyDocument=json.dumps(trust_policy),
    Description="Role for Bedrock Knowledge Bases to access resources"
)
role_arn = assume_role_response['Role']['Arn']

# Attach policies for S3 and OpenSearch access
iam.attach_role_policy(
    RoleName="bedrock-kb-access-role",
    PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
)

# Create inline policy for OpenSearch Serverless access
open_search_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {"Effect": "Allow", "Action": [
            "aoss:APIAccessAll"
        ], "Resource": "*"}
    ]
}

iam.put_role_policy(
    RoleName="bedrock-kb-access-role",
    PolicyName="OpenSearchAccess",
    PolicyDocument=json.dumps(open_search_policy)
)

# ─────────────────────────────────────────────────────────────
# STEP 2: Create OpenSearch Serverless Collection (Vector Store)
# ─────────────────────────────────────────────────────────────

aoss = boto3.client('opensearchserverless', region_name='us-east-1')

collection = aoss.create_collection(
    name="bedrock-kb-collection",
    type="VECTORSEARCH",
    standby_replicas="ENABLED",
)

collection_arn = collection['arn']
collection_id = collection['id']

# Wait for collection to be active
while True:
    status = aoss.get_collection(id=collection_id)['status']
    if status == 'ACTIVE':
        break
    print(f"Collection status: {status}, waiting...")
    time.sleep(30)

# Create access policy for the collection
access_policy = aoss.create_access_policy(
    name="bedrock-kb-access-policy",
    policy=json.dumps([
        {
            "Rules": [
                {"ResourceType": "collection", "Resource": [f"collection/{collection_id}"]},
                {"ResourceType": "index", "Resource": [f"index/{collection_id}/*"]},
            ],
            "Principal": {"AWS": [role_arn]},
            "Permission": [
                "aoss:CreateIndex",
                "aoss:DeleteIndex", 
                "aoss:UpdateIndex",
                "aoss:DescribeIndex",
                "aoss:ReadIndex",
                "aoss:WriteIndex",
            ]
        }
    ]),
    type="data"
)

# ─────────────────────────────────────────────────────────────
# STEP 3: Create Knowledge Base
# ─────────────────────────────────────────────────────────────

kb_response = bedrock.create_knowledge_base(
    name="company-docs-kb",
    description="Knowledge base for company documentation",
    roleArn=role_arn,
    knowledgeBaseConfiguration={
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": "arn:aws:bedrock:us-east-1::embedding-model/amazon.titan-embed-text-v2:0",
            "embeddingModelConfiguration": {
                "bedrockEmbeddingModelConfiguration": {
                    "dimensions": 1536,
                    "embeddingType": "SEMANTIC"
                }
            }
        }
    },
    storageConfiguration={
        "type": "OPENSEARCH_SERVERLESS",
        "opensearchServerlessConfiguration": {
            "collectionArn": collection_arn,
            "vectorIndexName": "bedrock-kb-index",
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "metadata"
            }
        }
    }
)

kb_arn = kb_response['knowledgeBase']['knowledgeBaseArn']
kb_id = kb_response['knowledgeBase']['knowledgeBaseId']

print(f"Knowledge Base created: {kb_arn}")

# ─────────────────────────────────────────────────────────────
# STEP 4: Create S3 Data Source
# ─────────────────────────────────────────────────────────────

s3_bucket_arn = "arn:aws:s3:::my-company-docs-bucket"

data_source_response = bedrock.create_data_source(
    knowledgeBaseId=kb_id,
    name="company-docs-source",
    dataSourceConfiguration={
        "type": "S3",
        "s3Configuration": {
            "bucketArn": s3_bucket_arn,
            "bucketOwnerAccountId": "123456789012",  # Required for cross-account
            "inclusionPrefixes": ["documents/", "policies/"],  # Optional filter
            # "exclusionPatterns": ["*.tmp", "drafts/*"],
        }
    },
    vectorIngestionConfiguration={
        "chunkingConfiguration": {
            "chunkingStrategy": "HIERARCHICAL",
            "hierarchicalChunkingConfiguration": {
                "levelConfigurations": [
                    {"maxTokens": 500, "overlapPercentage": 10},
                    {"maxTokens": 2000, "overlapPercentage": 5},
                ],
                "overlapTokens": 50
            }
        },
        "nativeIndexConfiguration": {
            "opensearchIndexName": "bedrock-kb-index"
        }
    }
)

ds_id = data_source_response['dataSource']['dataSourceId']
print(f"Data Source created: {ds_id}")
```

### 8.3.3 Run Initial Sync

```python
# ─────────────────────────────────────────────────────────────
# STEP 5: Start Sync Job
# ─────────────────────────────────────────────────────────────

sync_response = bedrock.start_ingestion_job(
    knowledgeBaseId=kb_id,
    dataSourceConfiguration={
        "dataSourceId": ds_id
    }
)

job_id = sync_response['ingestionJob']['jobId']
print(f"Sync job started: {job_id}")

# Monitor sync status
while True:
    status_response = bedrock.get_ingestion_job(
        knowledgeBaseId=kb_id,
        jobId=job_id
    )
    status = status_response['ingestionJob']['status']
    stats = status_response['ingestionJob'].get('statistics', {})
    
    print(f"Status: {status}")
    print(f"  Documents scanned: {stats.get('documentsScanned', 0)}")
    print(f"  Documents uploaded: {stats.get('documentsUploaded', 0)}")
    print(f"  Documents deleted: {stats.get('documentsDeleted', 0)}")
    print(f"  Documents failed: {stats.get('documentsFailed', 0)}")
    
    if status in ['COMPLETE', 'FAILED']:
        break
    time.sleep(30)

if status == 'FAILED':
    failure_msg = status_response['ingestionJob'].get('failureReason', 'Unknown')
    print(f"Sync failed: {failure_msg}")
```

## 8.4 Supported Data Sources

### 8.4.1 Amazon S3

```python
s3_config = {
    "type": "S3",
    "s3Configuration": {
        "bucketArn": "arn:aws:s3:::bucket-name",
        "bucketOwnerAccountId": "123456789012",  # Cross-account only
        "inclusionPrefixes": ["docs/", "policies/"],
        # "exclusionPatterns": ["*.tmp", "**/drafts/**"],
    }
}
```

**Supported file types:** PDF, HTML, Markdown, Plain text, CSV, JSON, Microsoft Word (DOCX), PowerPoint (PPTX)

**S3 permissions required:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"], "Resource": "arn:aws:s3:::bucket-name/*"},
    {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::bucket-name"}
  ]
}
```

### 8.4.2 Confluence

```python
confluence_config = {
    "type": "CONFLUENCE",
    "confluenceConfiguration": {
        "confluenceHostUrl": "https://company.atlassian.net/wiki",
        "confluenceSpaceIdentifier": ["ENG", "PRODUCT"],  # Space keys
        "pageCursor": "PAGE_CURSOR_VALUE",  # For pagination
        "excludeSub域": False,
        "crawlArchivedSpaces": False,
        "includePageAttachments": True,
        "filterConfiguration": {
            "objectType": {"value": "PAGE", "comparisonOperator": "CONTAINS"},
            "includeSubtypes": True
        }
    }
}
```

**Required secrets (stored in AWS Secrets Manager):**
- `confluence-username`: Confluence API username (email)
- `confluence-api-token`: Confluence API token

### 8.4.3 SharePoint Online

```python
sharepoint_config = {
    "type": "SHAREPOINT",
    "sharePointConfiguration": {
        "sharePointTenantUrl": "https://company.sharepoint.com",
        "siteCollIdentifier": ["site1.sharepoint.com/sites/Engineering"],
        "documentLibrary": ["Shared Documents", "Documents"],
        "excludeSharedByGuestUsers": False,
        "crawlComments": False,
        "crawlOneNoteFiles": False,
        "crawlSubSites": True,
        "enableFingerprinting": True,
    }
}
```

**Required secrets:**
- `sharepoint-client-id`: Azure AD app registration client ID
- `sharepoint-client-secret`: Azure AD app registration client secret

### 8.4.4 Salesforce

```python
salesforce_config = {
    "type": "SALESFORCE",
    "salesforceConfiguration": {
        "salesforceHostUrl": "https://company.my.salesforce.com",
        "salesforceObjectType": ["KnowledgeArticle", "Case"],
        "crawlAttachments": True,
        "filterConfiguration": {
            "documentTitleField": "Title",
            "fieldMappings": [
                {"sourceFieldName": "Title", "destinationFieldName": "title"},
                {"sourceFieldName": "Body", "destinationFieldName": "body"},
                {"sourceFieldName": "LastModifiedDate", "destinationFieldName": "last_modified"},
            ]
        }
    }
}
```

### 8.4.5 Web Crawler

```python
web_crawler_config = {
    "type": "WEB",
    "webConfiguration": {
        "urlConfiguration": {
            "seedUrlConfiguration": {
                "url": "https://docs.company.com",
                # OR crawlerUrls to specify specific URLs
            },
            "siteMapsConfiguration": {
                "siteMapUrl": "https://docs.company.com/sitemap.xml"
            }
        },
        "crawlerLimits": {
            "rateLimit": 5,  # Pages per second
            "maxCrawlDuration": 3600,  # Seconds
        },
        "scope": {
            "type": "HOST",  # or "SUBDOMAIN"
            "value": "docs.company.com"
        }
    }
}
```

## 8.5 Supported Embedding Models

| Model | Dimensions | Max Input | Languages | Bedrock Model ID |
|-------|-----------|-----------|-----------|-----------------|
| **Titan Embeddings G1 - Text v2** | 1536 | 8K tokens | 25+ | `amazon.titan-embed-text-v2:0` |
| **Titan Embeddings G1 - Text v1** | 4096 | 8K tokens | 25+ | `amazon.titan-embed-text-v1:0` |
| **Titan Embeddings G1 - Multimodal** | 1024 | 8K tokens + images | 25+ | `amazon.titan-embed-image-v1:0` |
| **Cohere Embed - English v3** | 1024 | 512 tokens | English | `cohere.embed-english-v3:0` |
| **Cohere Embed - Multilingual v3** | 1024 | 512 tokens | 100+ | `cohere.embed-multilingual-v3:0` |

### Titan Embeddings v2 Configuration

```python
embedding_config = {
    "embeddingModelArn": "arn:aws:bedrock:us-east-1::embedding-model/amazon.titan-embed-text-v2:0",
    "embeddingModelConfiguration": {
        "bedrockEmbeddingModelConfiguration": {
            "dimensions": 1536,  # Can reduce to save storage (Matryoshka)
            "embeddingType": "SEMANTIC"
        }
    }
}

# Dimension reduction without retraining (Matryoshka-style)
# 1536 → 1024 → 768 → 512 (lower dims = less storage, slightly lower quality)
# Supported: 1536, 1024, 768, 512, 256
```

## 8.6 Supported Vector Stores

### 8.6.1 Comparison Table

| Vector Store | Type | HNSW Support | Filtering | Best For |
|-------------|------|-------------|-----------|----------|
| **OpenSearch Serverless** | AWS SaaS | Yes (faiss) | Advanced | Default, AWS-native |
| **Aurora PostgreSQL** | AWS RDS | Yes | SQL-based | Existing Aurora infra |
| **Pinecone** | SaaS | Yes (proprietary) | Yes | Fully managed, global |
| **Redis** | Self-hosted/SaaS | Yes | Limited | Ultra-low latency |
| **MongoDB Atlas** | SaaS | Yes | Yes | MongoDB users |

### 8.6.2 OpenSearch Serverless (Default)

```python
opensearch_config = {
    "type": "OPENSEARCH_SERVERLESS",
    "opensearchServerlessConfiguration": {
        "collectionArn": "arn:aws:aoss:us-east-1:123456789:collections/xxx",
        "vectorIndexName": "bedrock-kb-index",
        "fieldMapping": {
            "vectorField": "vector",
            "textField": "text",
            "metadataField": "metadata"
        }
    }
}
```

### 8.6.3 Aurora PostgreSQL

```python
aurora_config = {
    "type": "AURORA",
    "auroraConfiguration": {
        "resourceArn": "arn:aws:rds:us-east-1:123456789:cluster:kb-aurora",
        "secretArn": "arn:aws:secretsmanager:us-east-1:123456789:secret:aurora/kb-secret",
        "databaseName": "bedrock_kb",
        "tableName": "bedrock_kb_chunks",
        "fieldMapping": {
            "vectorField": "embedding",
            "textField": "chunk_text",
            "metadataField": "chunk_metadata"
        }
    }
}
```

### 8.6.4 Pinecone

```python
pinecone_config = {
    "type": "PINECONE",
    "pineconeConfiguration": {
        "connectionString": "https://xxx.pinecone.io",
        "secretArn": "arn:aws:secretsmanager:us-east-1:123456789:secret:pinecone/api-key",
        "indexName": "bedrock-kb-index",
        "fieldMapping": {
            "vectorField": "values",
            "textField": "text",
            "metadataField": "metadata"
        }
    }
}
```

## 8.7 Chunking Configuration in Bedrock KB

Bedrock KB supports four chunking strategies. See Part 3, Section 3.11 for detailed explanations of each.

```python
# Strategy 1: None (entire document as one chunk)
chunking_none = {
    "chunkingStrategy": "NONE"
}

# Strategy 2: Fixed Size
chunking_fixed = {
    "chunkingStrategy": "FIXED_SIZE",
    "fixedSizeChunkingConfiguration": {
        "maxTokens": 500,
        "overlapPercentage": 20
    }
}

# Strategy 3: Hierarchical (Recommended)
chunking_hierarchical = {
    "chunkingStrategy": "HIERARCHICAL",
    "hierarchicalChunkingConfiguration": {
        "levelConfigurations": [
            {"maxTokens": 500, "overlapPercentage": 10},
            {"maxTokens": 2000, "overlapPercentage": 5},
        ],
        "overlapTokens": 50
    }
}

# Strategy 4: Semantic
chunking_semantic = {
    "chunkingStrategy": "SEMANTIC",
    "semanticChunkingConfiguration": {
        "maxTokens": 500,
        "breakpointPercentileThreshold": 95,
        "bufferSize": 1
    }
}
```

### Recommended Chunking by Use Case

| Use Case | Strategy | Config |
|----------|----------|--------|
| General Q&A | Hierarchical | 500 → 2000 tokens, 10% overlap |
| Long documents | Hierarchical | 300 → 1500 tokens |
| Code search | Fixed Size | 256 tokens, 25% overlap |
| Short policies | None or Fixed | 1000 tokens, 10% overlap |
| Multi-topic docs | Semantic | 500 tokens, threshold 90 |

## 8.8 The `RetrieveAndGenerate` API — Deep Dive

This is the primary API for RAG with Bedrock Knowledge Bases. It handles retrieval + generation in one call.

### 8.8.1 API Parameters

```python
response = bedrock_agent_runtime.retrieve_and_generate(
    input={
        "text": "How do I reset my AWS root account password?"
    },
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": kb_id,
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-4-20250514",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 10,  # k for retrieval
                    "overrideSearchType": "HYBRID",  # or "SEMANTIC", "HYBRID"
                    "filter": {
                        "andAll": [
                            {
                                "key": "document_type",
                                "value": "policy",
                                "operator": "EQ"
                            },
                            {
                                "key": "department",
                                "value": "engineering",
                                "operator": "EQ"
                            }
                        ]
                    }
                }
            },
            "generationConfiguration": {
                "promptTemplate": {
                    "textPromptTemplate": """{{#if $context}}$system_instruction\n\nContext:\n$context\n\n---\n{{/if}}\n\nQuestion: $question\n\nAnswer:"""
                },
                "inferenceConfiguration": {
                    "maxTokens": 2048,
                    "temperature": 0.3,
                    "topP": 0.9,
                    "stopSequences": ["\n\nHuman:"]
                }
            }
        }
    }
)
```

### 8.8.2 Response Structure

```python
{
    "output": {
        "text": "To reset your AWS root account password, follow these steps:\n\n1. Sign in to the AWS Management Console...",
        "citations": [
            {
                "generatedResponsePart": {
                    "textSegment": "To reset your AWS root account password...",
                    "sourceReference": {
                        "sourceId": "c2e6b2a1-...",
                        "location": {
                            "type": "CHUNK",
                            "score": 0.847,
                            "offset": {
                                "start": 0,
                                "end": 250
                            },
                            "document": {
                                "uri": "s3://bucket/prefix/policy.pdf",
                                "title": "AWS Account Security Policy"
                            }
                        }
                    }
                }
            }
        ]
    },
    "retrievalDetails": {
        "retrievalIds": ["chunk-id-1", "chunk-id-2"],
        "knowledgeBaseId": "kb-12345",
        "retrievalTimestamp": "2025-01-15T10:30:00Z"
    }
}
```

### 8.8.3 Full Example with Error Handling

```python
import boto3
from botocore.exceptions import ClientError

bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

class BedrockKBClient:
    def __init__(self, kb_id: str, model_id: str = "anthropic.claude-3-sonnet-4-20250514"):
        self.kb_id = kb_id
        self.model_id = model_id
        self.client = bedrock_runtime
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        metadata_filter: dict = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict:
        """
        Query the knowledge base with RAG.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            metadata_filter: Optional filter on chunk metadata
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
        
        Returns:
            dict with answer, citations, and metadata
        """
        try:
            # Build filter expression
            retrieval_config = {
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "SEMANTIC",
                }
            }
            
            # Add filter if provided
            if metadata_filter:
                retrieval_config["vectorSearchConfiguration"]["filter"] = self._build_filter(metadata_filter)
            
            response = self.client.retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.kb_id,
                        "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{self.model_id}",
                        "retrievalConfiguration": retrieval_config,
                        "generationConfiguration": {
                            "inferenceConfiguration": {
                                "maxTokens": max_tokens,
                                "temperature": temperature,
                                "topP": 0.9,
                            }
                        }
                    }
                }
            )
            
            return self._parse_response(response)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid request: {error_msg}")
            elif error_code == 'ResourceNotFoundException':
                raise KeyError(f"Knowledge base {self.kb_id} not found")
            elif error_code == 'InternalServerException':
                raise RuntimeError(f"Bedrock server error: {error_msg}")
            else:
                raise
    
    def _build_filter(self, filter_dict: dict) -> dict:
        """Convert Python filter dict to Bedrock filter format."""
        clauses = []
        for key, value in filter_dict.items():
            clauses.append({
                "key": key,
                "value": str(value),
                "operator": "EQ"
            })
        
        if len(clauses) == 1:
            return clauses[0]
        return {"andAll": clauses}
    
    def _parse_response(self, response: dict) -> dict:
        """Parse and normalize the API response."""
        output = response.get('output', {})
        citations = []
        
        for cit in output.get('text', ''):
            pass  # Already in text
        
        # Extract citations
        raw_citations = response.get('citations', [])
        for i, cit in enumerate(raw_citations):
            location = cit.get('location', {})
            citations.append({
                "source_num": i + 1,
                "text": cit.get('generatedResponsePart', {}).get('textSegment', ''),
                "document_title": location.get('document', {}).get('title', 'Unknown'),
                "document_uri": location.get('document', {}).get('uri', ''),
                "score": location.get('score', 0),
            })
        
        return {
            "answer": output.get('text', ''),
            "citations": citations,
            "retrieval_details": response.get('retrievalDetails', {}),
        }

# Usage
client = BedrockKBClient(kb_id="kb-12345")
result = client.query(
    question="How do I reset my AWS root account password?",
    top_k=10,
    metadata_filter={"document_type": "policy"},
)

print(result["answer"])
print("\nSources:")
for cit in result["citations"]:
    print(f"[{cit['source_num']}] {cit['document_title']} (score: {cit['score']:.3f})")
```

## 8.9 The `Retrieve` API — Building Custom RAG Pipelines

Use the `Retrieve` API when you want to build a custom RAG pipeline with custom generation logic.

```python
def retrieve_from_kb(
    client,
    kb_id: str,
    query: str,
    top_k: int = 10,
    metadata_filter: dict = None,
) -> list[dict]:
    """
    Retrieve relevant chunks from Bedrock Knowledge Base.
    Use this for custom RAG pipelines where you control generation.
    """
    
    retrieval_config = {
        "vectorSearchConfiguration": {
            "numberOfResults": top_k,
        }
    }
    
    if metadata_filter:
        retrieval_config["vectorSearchConfiguration"]["filter"] = build_filter(metadata_filter)
    
    response = client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalConfiguration=retrieval_config,
        retrievalQuery={
            "text": query
        }
    )
    
    results = []
    for result in response.get('retrievalResults', []):
        results.append({
            "content": result['content']['text'],
            "score": result['score'],
            "metadata": result.get('metadata', {}),
            "location": {
                "uri": result.get('location', {}).get('uri', ''),
                "title": result.get('location', {}).get('title', ''),
            }
        })
    
    return results

# Full custom RAG with Retrieve API
def custom_rag_pipeline(question: str, kb_id: str):
    """
    Custom RAG pipeline:
    1. Retrieve chunks from Bedrock KB
    2. Post-process and filter chunks
    3. Apply custom prompt engineering
    4. Call LLM directly
    """
    
    # Step 1: Retrieve
    chunks = retrieve_from_kb(
        client=bedrock_runtime,
        kb_id=kb_id,
        query=question,
        top_k=20,  # Over-retrieve for re-ranking
    )
    
    # Step 2: Filter by score threshold
    relevant_chunks = [c for c in chunks if c['score'] > 0.7]
    
    # Step 3: Assemble context
    context = "\n\n---\n\n".join([
    f"[Source: {c['location']['title']}]\n{c['content']}"
    for c in relevant_chunks[:5]
])
    
    # Step 4: Custom prompt
    prompt = f"""You are a helpful assistant. Use ONLY the following context to answer.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer: """
    
    # Step 5: Call LLM directly
    llm = boto3.client('bedrock-runtime')
    response = llm.invoke_model(
        modelId='anthropic.claude-3-sonnet-4-20250514',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )
    
    return json.loads(response['body'].read())
```

## 8.10 Sync Jobs: Manual, Scheduled, Event-Driven

### 8.10.1 Manual Sync

```python
# Start one-time sync
response = bedrock.start_ingestion_job(
    knowledgeBaseId=kb_id,
    dataSourceConfiguration={
        "dataSourceId": ds_id
    }
)
job_id = response['ingestionJob']['jobId']

# Get job status
status = bedrock.get_ingestion_job(
    knowledgeBaseId=kb_id,
    jobId=job_id
)
print(f"Status: {status['ingestionJob']['status']}")
```

### 8.10.2 Scheduled Sync via EventBridge

```python
import boto3
events = boto3.client('events')

# Create EventBridge rule for daily sync at 2 AM UTC
rule = events.put_rule(
    Name="bedrock-kb-daily-sync",
    ScheduleExpression="cron(0 2 * * ? *)",  # Every day at 2 AM UTC
    State="ENABLED",
    Description="Trigger Bedrock KB sync daily"
)

# Add Bedrock as target (using StartIngestionJob API via Lambda)
lambda_func_arn = "arn:aws:lambda:us-east-1:123456789:function:bedrock-kb-sync"

events.put_targets(
    Rule="bedrock-kb-daily-sync",
    Targets=[{
        "Id": "bedrock-kb-sync-target",
        "Arn": lambda_func_arn,
        "Input": json.dumps({"kb_id": kb_id, "ds_id": ds_id})
    }]
)

# Lambda function to trigger sync
def lambda_handler(event, context):
    kb_id = event['kb_id']
    ds_id = event['ds_id']
    
    bedrock_client = boto3.client('bedrock')
    response = bedrock_client.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceConfiguration={"dataSourceId": ds_id}
    )
    
    return {
        "statusCode": 200,
        "jobId": response['ingestionJob']['jobId']
    }
```

### 8.10.3 Event-Driven Sync (S3 Triggers)

```python
# Create EventBridge rule for S3 object created events
s3_rule = events.put_rule(
    Name="bedrock-kb-s3-sync",
    EventPattern=json.dumps({
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "detail": {
            "bucket": {"name": ["my-company-docs-bucket"]},
            "object": {"key": [{"prefix": "documents/"}]}
        }
    }),
    State="ENABLED"
)

# Target Lambda to trigger sync
events.put_targets(
    Rule="bedrock-kb-s3-sync",
    Targets=[{
        "Id": "bedrock-kb-s3-target",
        "Arn": lambda_func_arn,
        "Input": json.dumps({"kb_id": kb_id, "ds_id": ds_id})
    }]
)
```

## 8.11 Ingestion Job Monitoring and Troubleshooting

### 8.11.1 Common Failure Modes

| Status | Meaning | Resolution |
|--------|---------|------------|
| `COMPLETE` | Success | — |
| `FAILED` | General failure | Check `failureReason` field |
| `PARTIAL_SUCCESS` | Some documents failed | Check `statistics.documentsFailed` |
| `IN_PROGRESS` | Running | Wait and monitor |

### 8.11.2 Diagnosing Failures

```python
def diagnose_sync_failure(kb_id: str, job_id: str):
    """Diagnose why a sync job failed."""
    
    response = bedrock.get_ingestion_job(
        knowledgeBaseId=kb_id,
        jobId=job_id
    )
    
    job = response['ingestionJob']
    
    print("=== Sync Job Diagnostics ===")
    print(f"Status: {job['status']}")
    print(f"Started: {job.get('startTime')}")
    print(f"Updated: {job.get('lastUpdatedAt')}")
    
    if 'statistics' in job:
        stats = job['statistics']
        print(f"\n=== Statistics ===")
        print(f"Scanned: {stats.get('documentsScanned', 0)}")
        print(f"Uploaded: {stats.get('documentsUploaded', 0)}")
        print(f"Deleted: {stats.get('documentsDeleted', 0)}")
        print(f"Failed: {stats.get('documentsFailed', 0)}")
    
    if job['status'] == 'FAILED':
        print(f"\n=== Failure Reason ===")
        print(job.get('failureReason', 'Unknown'))
        
        # Check CloudWatch for detailed logs
        logs_client = boto3.client('logs')
        
        # Find KB-related log groups
        log_groups = logs_client.describe_log_groups(
            logGroupNamePattern="/aws/bedrock"
        )
        
        print("\n=== CloudWatch Logs ===")
        for lg in log_groups['logGroups']:
            print(f"Log Group: {lg['logGroupName']}")
```

### 8.11.3 Common Issues and Solutions

```python
# Issue 1: Access denied to S3 bucket
# Solution: Ensure IAM role has s3:GetObject permission on the bucket

# Issue 2: Unsupported file format
# Solution: Convert files to supported formats (PDF, HTML, MD, TXT, DOCX, JSON, CSV)

# Issue 3: Document too large
# Solution: Split large documents before ingestion, or use higher chunking overlap

# Issue 4: Characters not extracted from PDF
# Solution: Use Amazon Textract for better PDF text extraction (configure in KB)

# Issue 5: OpenSearch collection not active
# Solution: Wait for collection to be ACTIVE before creating KB

# Issue 6: Cross-account S3 bucket access denied
# Solution: Ensure bucket policy allows GetObject from KB role's account

bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::KB_ACCOUNT_ID:role/bedrock-kb-access-role"},
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::bucket-name",
                "arn:aws:s3:::bucket-name/*"
            ]
        }
    ]
}
```

## 8.12 Knowledge Base Versioning

Bedrock KB doesn't have built-in versioning, but you can implement it:

```python
class KnowledgeBaseVersioning:
    """
    Implement versioning for Bedrock KB by tagging chunks with version metadata.
    """
    
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
    
    def index_with_version(self, chunks: list, version: str):
        """
        Index documents with version metadata.
        """
        for chunk in chunks:
            chunk.metadata['version'] = version
            chunk.metadata['indexed_at'] = datetime.now().isoformat()
        
        # Index via custom ingestion (pre-chunk and embed)
        self._custom_ingest(chunks)
    
    def query_latest_only(self, query: str, kb_client: BedrockKBClient):
        """
        Query only the latest version of documents.
        """
        # First get all versions
        all_chunks = retrieve_from_kb(kb_id, query, top_k=100)
        
        # Group by document_id and keep only latest
        by_doc = {}
        for chunk in all_chunks:
            doc_id = chunk['metadata'].get('document_id')
            version = chunk['metadata'].get('version', '0')
            
            if doc_id not in by_doc or version > by_doc[doc_id]['version']:
                by_doc[doc_id] = chunk
        
        return list(by_doc.values())
    
    def delete_old_version(self, version: str):
        """
        Delete chunks from old version.
        Note: Bedrock KB doesn't support direct delete by metadata.
        Workaround: Use custom vector store and track chunk IDs.
        """
        # For full version control, use custom ingestion pipeline
        # that tracks chunk_ids externally
        pass
```

## 8.13 Cross-Region Considerations

### 8.13.1 Architecture Options

| Option | Description | Use Case |
|--------|-------------|----------|
| Single region | All resources in one region | Low latency, data residency |
| Cross-region sync | Primary KB in one region, read replicas | Disaster recovery |
| Regional KBs | Separate KBs per region, same data source | Data residency compliance |

### 8.13.2 Data Residency Implementation

```python
def create_regional_kbs(account_id: str, regions: list[str]):
    """
    Create separate KB per region for data residency.
    All KBs read from the same S3 bucket (with proper region config).
    """
    kbs = {}
    
    for region in regions:
        bedrock = boto3.client('bedrock', region_name=region)
        
        kb = bedrock.create_knowledge_base(
            name=f"company-kb-{region}",
            description=f"Data residency: {region}",
            roleArn=f"arn:aws:iam::{account_id}:role/bedrock-kb-{region}",
            # ... rest of config
        )
        
        kbs[region] = kb['knowledgeBase']['knowledgeBaseArn']
    
    return kbs

def route_to_regional_kb(user_region: str, question: str) -> dict:
    """
    Route user query to KB in their region for data residency.
    """
    kb_arn = regional_kbs.get(user_region)
    if not kb_arn:
        kb_arn = regional_kbs['us-east-1']  # Default fallback
    
    client = BedrockKBClient(kb_id=kb_arn.split('/')[-1])
    return client.query(question)
```

## 8.14 Bedrock KB Quotas and Limits

| Resource | Default Limit | Can Request Increase? |
|----------|--------------|----------------------|
| Knowledge Bases per region | 50 | Yes |
| Data sources per KB | 10 | Yes |
| Documents per KB | 1,000,000 | Yes |
| Max document size | 50 MB | No |
| Max chunks per document | 10,000 | Yes |
| Max embedding dimensions | 16,384 | No |
| Max retrieval results (k) | 100 | Yes |
| Concurrent sync jobs | 5 per KB | Yes |
| `RetrieveAndGenerate` TPS | 100 per account | Yes |
| `Retrieve` TPS | 100 per account | Yes |

### Requesting Limit Increases

```python
# Via AWS Support API
support = boto3.client('support', region_name='us-east-1')

response = support.create_case(
    subject="Bedrock KB Limit Increase Request",
    serviceCode="bedrock",
    severityCode="normal",
    categoryCode="limit-increase",
    communicationBody=f"""
    Requesting increase for Bedrock Knowledge Bases:
    
    Limit type: Knowledge Bases per region
    Current limit: 50
    Requested limit: 100
    
    Use case: [describe your use case]
    """,
    issueType="technical-question"
)
```

## 8.15 Cost Model and Optimization

### 8.15.1 Cost Components

| Component | Pricing (us-east-1) | Notes |
|-----------|---------------------|-------|
| **Knowledge Base storage** | $0.024/GB/month | Stored in your vector store |
| **OpenSearch Serverless storage** | $0.24/GB/month | Vector data in OpenSearch |
| **OpenSearch Serverless compute** | $0.024/OCU-hour | ~2 vCPU + 8GB |
| **Embedding inference** | Titan v2: $0.0002/1K tokens | Batch embedding |
| **RetrieveAndGenerate** | Model-specific | Claude 3.5 Sonnet: $0.003/1K tokens |
| **Sync job** | Free | No per-job charge |

### 8.15.2 Cost Estimation

```python
def estimate_monthly_cost(
    num_documents: int,
    avg_doc_chars: int,
    avg_chunks_per_doc: int,
    daily_queries: int,
    avg_query_chars: int,
    avg_response_tokens: int,
) -> dict:
    """
    Estimate monthly cost for a Bedrock KB deployment.
    """
    
    # Storage calculations
    chars_per_chunk = avg_doc_chars / avg_chunks_per_doc
    chunks_total = num_documents * avg_chunks_per_doc
    
    # Titan v2 embedding: ~1 token per 4 chars
    tokens_per_chunk = chars_per_chunk / 4
    embedding_tokens = chunks_total * tokens_per_chunk
    
    # Embedding cost ($0.0002 per 1K tokens)
    embedding_cost = (embedding_tokens / 1000) * 0.0002
    
    # Query embeddings (one per query)
    daily_query_tokens = daily_queries * (avg_query_chars / 4)
    monthly_query_tokens = daily_query_tokens * 30
    query_embedding_cost = (monthly_query_tokens / 1000) * 0.0002
    
    # Generation cost (Claude 3.5 Sonnet example)
    # Input: query + context tokens, Output: response tokens
    avg_context_tokens = 10 * tokens_per_chunk  # 10 retrieved chunks
    input_tokens_per_query = (avg_query_chars / 4) + avg_context_tokens
    output_tokens_per_query = avg_response_tokens
    
    input_cost = (input_tokens_per_query * daily_queries * 30 / 1000) * 0.003
    output_cost = (output_tokens_per_query * daily_queries * 30 / 1000) * 0.015
    
    # OpenSearch storage (1536 floats = 6KB per chunk)
    vector_storage_gb = (chunks_total * 6) / (1024 * 1024 * 1024)
    opensearch_storage = vector_storage_gb * 0.24
    
    # OpenSearch compute (estimate 2 OCU for moderate load)
    opensearch_compute = 2 * 24 * 30 * 0.024
    
    return {
        "embedding_cost": embedding_cost,
        "query_embedding_cost": query_embedding_cost,
        "generation_input_cost": input_cost,
        "generation_output_cost": output_cost,
        "opensearch_storage": opensearch_storage,
        "opensearch_compute": opensearch_compute,
        "total_monthly": (
            embedding_cost + query_embedding_cost + 
            input_cost + output_cost + 
            opensearch_storage + opensearch_compute
        )
    }

# Example: 10K documents, 5000 chars each, 10 chunks/doc
# 1000 daily queries
estimate = estimate_monthly_cost(
    num_documents=10000,
    avg_doc_chars=5000,
    avg_chunks_per_doc=10,
    daily_queries=1000,
    avg_query_chars=200,
    avg_response_tokens=500,
)
print(estimate)
# {'total_monthly': ~$350-500/month range}
```

### 8.15.3 Cost Optimization Strategies

1. **Reduce embedding dimensions**: Titan v2 supports 1536→1024→768→512 reduction
2. **Use smaller chunks**: More chunks but lower per-chunk embedding cost; but balance with retrieval quality
3. **Implement caching**: Cache frequently asked queries with their retrieved results
4. **Reduce retrieval k**: Default 10 may be excessive; tune based on query complexity
5. **Use smaller models when appropriate**: Claude Haiku for simple Q&A vs Sonnet for complex analysis
6. **Schedule sync jobs off-peak**: OpenSearch Serverless auto-scales, but steady state is cheaper

## 8.16 IAM Permissions for Bedrock KB

### 8.16.1 Minimal IAM Policy for KB with S3 Source

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateKnowledgeBase",
        "bedrock:DeleteKnowledgeBase",
        "bedrock:GetKnowledgeBase",
        "bedrock:ListKnowledgeBases",
        "bedrock:CreateDataSource",
        "bedrock:DeleteDataSource",
        "bedrock:StartIngestionJob",
        "bedrock:GetIngestionJob",
        "bedrock:ListIngestionJobs",
        "bedrock:Retrieve",
        "bedrock:RetrieveAndGenerate"
      ],
      "Resource": "arn:aws:bedrock:us-east-1:123456789:knowledge-base/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "aoss:APIAccessAll"
      ],
      "Resource": "*"  # Scope to specific collection ARN in production
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:123456789:secret:*"
    }
  ]
}
```

### 8.16.2 Service-Linked Role for Bedrock

```python
# Create service-linked role (automatically created on first KB creation)
iam_client = boto3.client('iam')

# Bedrock creates this automatically, but for explicit creation:
slr = iam_client.create_service_linked_role(
    AWSServiceName='bedrock.amazonaws.com',
    Description='Service-linked role for Amazon Bedrock'
)
```

## 8.17 VPC and Networking for Bedrock KB

Bedrock KB supports VPC connectivity for private data access:

```python
# Create VPC endpoint for Bedrock

# ForRetrieveAndGenerate API (runtime)
ec2 = boto3.client('ec2')

vpce = ec2.create_vpc_endpoint(
    VpcId='vpc-12345',
    ServiceName='com.amazonaws.us-east-1.bedrock-agent-runtime',
    VpcEndpointType='Interface',
    SubnetIds=['subnet-123', 'subnet-456'],
    SecurityGroupIds=['sg-12345'],
    PrivateDnsEnabled=True,
)

# For KB management (control plane)
vpce_control = ec2.create_vpc_endpoint(
    VpcId='vpc-12345',
    ServiceName='com.amazonaws.us-east-1.bedrock',
    VpcEndpointType='Interface',
    SubnetIds=['subnet-123', 'subnet-456'],
    SecurityGroupIds=['sg-12345'],
    PrivateDnsEnabled=True,
)
```

### Security Group Requirements

```json
{
  "InboundRules": [
    {"Protocol": "TCP", "FromPort": 443, "ToPort": 443, "Source": "vpc-cidr"},
  ],
  "OutboundRules": [
    {"Protocol": "all", "Destination": "0.0.0.0/0"},
  ]
}
```

---

---

# Part 9: Amazon Titan Embeddings — Deep Dive

Titan Embeddings is AWS's native embedding model family, deeply integrated with Bedrock. Understanding its capabilities and behavior is essential for optimized Bedrock KB implementations.

## 9.1 Titan Embeddings Model Versions

### 9.1.1 Titan Embeddings G1 — Text v2 (Recommended)

The current recommended model for most RAG workloads:

| Property | Value |
|----------|-------|
| **Model ID** | `amazon.titan-embed-text-v2:0` |
| **Dimensions** | 1536 (configurable: 1536, 1024, 768, 512, 256) |
| **Max Input** | 8K tokens |
| **Output** | L2-normalized dense vectors |
| **Languages** | 25+ (English, Spanish, French, German, Portuguese, Italian, Dutch, Chinese, Japanese, Korean, Arabic, etc.) |
| **Embedding Type** | SEMANTIC (semantic similarity) |
| **Normalization** | L2-normalized (unit length) |

**Key innovation — Matryoshka Dimension Reduction:** Unlike other embedding models with fixed dimensions, Titan v2 supports truncating to lower dimensions without retraining. This allows storage/performance trade-offs post-embedding.

### 9.1.2 Titan Embeddings G1 — Text v1 (Legacy)

| Property | Value |
|----------|-------|
| **Model ID** | `amazon.titan-embed-text-v1:0` |
| **Dimensions** | 4096 (fixed) |
| **Max Input** | 8K tokens |
| **Languages** | 25+ |

**When to use:** Legacy deployments, specific requirements for 4096 dimensions.

### 9.1.3 Titan Embeddings G1 — Multimodal

```python
# Multi-modal embedding configuration
multimodal_config = {
    "modelArn": "arn:aws:bedrock:us-east-1::embedding-model/amazon.titan-embed-image-v1:0",
    "embeddingModelConfiguration": {
        "bedrockEmbeddingModelConfiguration": {
            "dimensions": 1024,
            "embeddingType": "MULTIMODAL"
        }
    }
}
```

Supports embedding image + text into the same vector space, enabling:
- Image search from text queries
- Text search from image queries
- Documents with figures, charts, and diagrams

## 9.2 Model Specifications Deep Dive

### 9.2.1 Tokenization and Context Handling

Titan uses a custom subword tokenizer optimized for multilingual text:
- ~1 token per 4 characters for English
- Higher compression for languages with compound words (German)
- Byte-pair encoding (BPE) variant trained on 25+ languages

**Handling long inputs (>8K tokens):**
- For inputs exceeding 8K tokens, manually split and embed sections
- Average or concatenate section embeddings (loss of some semantic nuance)
- For RAG, prefer to chunk to < 8K tokens before embedding (Bedrock KB does this automatically)

### 9.2.2 Embedding Output Format

```python
import boto3
import json

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

def embed_text_titan_v2(text: str, dimensions: int = 1536) -> list[float]:
    """
    Generate Titan v2 embeddings.
    Returns an L2-normalized 1536-dimensional vector by default.
    """
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({
            "inputText": text
        }),
        accept='application/json',
        contentType='application/json'
    )
    
    response_body = json.loads(response['body'].read().decode())
    
    return response_body['embedding']  # list of floats

def embed_batch_titan_v2(texts: list[str]) -> list[list[float]]:
    """
    Batch embed multiple texts.
    Titan v2 supports batch input for efficiency.
    """
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({
            "inputText": texts  # List of strings for batch
        }),
        accept='application/json',
        contentType='application/json'
    )
    
    response_body = json.loads(response['body'].read().decode())
    return response_body['embeddings']

# Example usage
single = embed_text_titan_v2("How do I reset my password?")
print(f"Single embedding dimensions: {len(single)}")  # 1536

batch = embed_batch_titan_v2([
    "How do I reset my password?",
    "How do I change my email?",
    "Where can I find my billing history?"
])
print(f"Batch embeddings: {len(batch)} x {len(batch[0])} dimensions")  # 3 x 1536
```

## 9.3 Normalization Behavior

Titan v2 outputs are **L2-normalized** to unit length (Euclidean norm = 1.0). This has important implications:

```python
import numpy as np

def verify_normalization(embedding: list[float]) -> bool:
    """
    Titan v2 embeddings should have L2 norm of 1.0.
    This enables using dot product as cosine similarity.
    """
    norm = np.linalg.norm(embedding)
    return abs(norm - 1.0) < 1e-6

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    For normalized vectors, dot product equals cosine similarity.
    No need for division by norms.
    """
    return np.dot(v1, v2)

def euclidean_to_cosine(euclidean_dist: float) -> float:
    """
    Convert Euclidean distance to cosine similarity.
    For normalized vectors: euclidean² = 2 × (1 - cosine)
    So cosine = 1 - euclidean²/2
    """
    return 1 - (euclidean_dist ** 2) / 2
```

**Why L2 normalization matters:**
- Dot product computation is faster than explicit cosine calculation
- Magnitude doesn't affect similarity (only direction matters)
- Consistent scoring across all vectors

## 9.4 Multilingual Capabilities

Titan v2 was trained on 25+ languages and handles:
- Cross-lingual retrieval (English query → French document)
- Multilingual documents (same document has multiple languages)
- Code-switched text (mixing languages in one document)

### Multilingual Benchmark Performance (MTEB)

| Language Group | Titan v2 Performance | Cohere Multilingual |
|---------------|---------------------|---------------------|
| English | ~62% (MTEB avg) | ~62% |
| Romance (FR, ES, IT, PT) | Good | Excellent |
| Germanic (DE, NL) | Good | Excellent |
| East Asian (ZH, JA, KO) | Moderate | Good |
| Arabic/Hebrew | Moderate | Good |

### Multilingual Query Example

```python
# Query in English retrieves documents in other languages
query_emb = embed_text_titan_v2("How to configure network settings?")

#检索 returns French document with high similarity
results = vector_store.search(query_emb, k=5)
# Retrieved: "Comment configurer les paramètres réseau..." (French)
# This works because the semantic meaning is preserved across languages
```

## 9.5 Code Embedding Performance

Titan v2 handles code reasonably well but is not specialized for code. For code-heavy RAG workloads:

| Aspect | Titan v2 | Code-Specific Models (e.g., GTE) |
|--------|----------|--------------------------------|
| Natural language | Excellent | Good |
| Python/JavaScript | Good | Excellent |
| SQL queries | Good | Excellent |
| Shell scripts | Moderate | Good |
| Config files (YAML, JSON) | Moderate | Good |
| Technical documentation | Good | Good |

### Code-Specific Optimization

```python
def preprocess_code_for_embedding(code: str, language: str) -> str:
    """
    Preprocess code to improve Titan embedding quality.
    """
    lines = code.split('\n')
    
    # Normalize indentation (remove leading whitespace variability)
    min_indent = min(
        (len(line) - len(line.lstrip())) for line in lines if line.strip()
    )
    normalized = '\n'.join(
        line[min_indent:] if line.strip() else '' for line in lines
    )
    
    # Add language context (helps semantic understanding)
    enhanced = f"[{language} code]\n{normalized}"
    
    return enhanced

# Example
code = """
    def authenticate_user(username, password):
        if validate_credentials(username, password):
            return generate_token(username)
        raise AuthError("Invalid credentials")
"""

processed = preprocess_code_for_embedding(code, "python")
embedding = embed_text_titan_v2(processed)
```

## 9.6 Dimension Reduction (Matryoshka)

Titan v2 supports post-hoc dimension reduction without retraining:

```python
def truncate_embedding(embedding: list[float], target_dim: int) -> list[float]:
    """
    Truncate Titan v2 embedding to lower dimensions.
    Uses first N dimensions (they're ordered by importance).
    
    Supported targets: 1536, 1024, 768, 512, 256
    """
    if target_dim > len(embedding):
        raise ValueError(f"Target {target_dim} > original {len(embedding)}")
    
    # Just take first N dimensions
    truncated = embedding[:target_dim]
    
    # Re-normalize to unit length (important!)
    norm = np.linalg.norm(truncated)
    return [x / norm for x in truncated]

# Storage comparison
original_1536 = embed_text_titan_v2("long document text...")
dim_512 = truncate_embedding(original_1536, 512)
dim_256 = truncate_embedding(original_1536, 256)

# Storage: 1536 × 4 bytes = 6KB → 512 × 4 = 2KB → 256 × 4 = 1KB
# Quality loss: ~2-5% recall degradation typical
```

### Dimension vs Quality Trade-off

| Dimensions | Storage Reduction | Recall Retention | Best For |
|------------|------------------|------------------|----------|
| 1536 | baseline | 100% | Maximum quality |
| 1024 | 33% reduction | ~97-99% | Balanced workloads |
| 768 | 50% reduction | ~95-97% | Storage-constrained |
| 512 | 67% reduction | ~90-95% | Large scale |
| 256 | 83% reduction | ~85-90% | Cost-sensitive |

## 9.7 Pricing and Throughput

### Pricing (us-east-1, as of 2025)

| Model | Input | Price per 1K tokens |
|-------|-------|--------------------|
| Titan Embeddings v2 | $0.0002 | — |
| Titan Embeddings v1 | $0.0001 | — |
| Titan Multimodal | $0.0002 | — |

**Cost calculation:**
```python
def calculate_embedding_cost(num_documents: int, avg_tokens_per_doc: int) -> float:
    """
    Calculate embedding cost for document ingestion.
    
    Example: 10,000 documents, avg 1000 tokens each
    """
    total_tokens = num_documents * avg_tokens_per_doc
    cost_per_1k = 0.0002  # Titan v2
    
    return (total_tokens / 1000) * cost_per_1k

# 10K docs × 1000 tokens = 10M tokens = $2.00
print(calculate_embedding_cost(10000, 1000))  # $2.00
```

### Throughput

- **Single-threaded**: ~50-100 embeddings/second (varies by input length)
- **Batch API**: Up to 25x throughput improvement
- **Bedrock KB managed**: Automatic batching and parallelization

## 9.8 Comparison with Other Embedding Models

| Model | Dimensions | Max Tokens | Languages | Strengths | Weaknesses |
|-------|-----------|-----------|-----------|----------|------------|
| **Titan v2** | 1536 | 8K | 25+ | AWS-native, cost-effective | Not best for code |
| **Titan v1** | 4096 | 8K | 25+ | Higher dimensions | Legacy, more storage |
| OpenAI text-embedding-3-large | 3072 | 8K | English-centric | Best quality EN | Expensive, EN-focused |
| OpenAI text-embedding-3-small | 1536 | 8K | English-centric | Cheap, fast | Non-English weaker |
| Cohere embed-english-v3 | 1024 | 512 | English | Efficient EN | Short context |
| Cohere embed-multilingual-v3 | 1024 | 512 | 100+ | Best multilingual | Short context |
| BGE-large-en-v1.5 | 1024 | 512 | English | Open source, good quality | Short context |

### When to Choose Titan v2

✅ **Choose Titan v2 when:**
- Already using AWS/Bedrock ecosystem
- Cost-sensitive (lowest embedding cost among comparable models)
- Want managed service without API key management
- Multilingual but not deeply specialized (25+ languages)
- Simple integration (no API key rotation needed)

❌ **Consider alternatives when:**
- Need absolute best quality (OpenAI text-embedding-3-large)
- Need specialized code embeddings (CodeBERT, GTE-code)
- Need >512 token context for embedding (Cohere limited)
- Using multi-cloud architecture

---

# Part 10: Building RAG with AWS — Code Examples

This section provides complete, working code examples for building RAG systems with AWS services.

## 10.1 Minimal RAG with Bedrock KB (boto3)

```python
import boto3
import json
from datetime import datetime

class MinimalBedrockRAG:
    """
    Minimal working RAG implementation using Bedrock Knowledge Bases.
    All you need: a created Knowledge Base with synced data.
    """
    
    def __init__(self, kb_id: str, model_id: str = "anthropic.claude-3-sonnet-4-20250514"):
        self.kb_id = kb_id
        self.model_id = model_id
        self.client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
    
    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        Ask a question and get an answer with citations.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
        
        Returns:
            dict with 'answer', 'sources', and 'metadata'
        """
        response = self.client.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": self.kb_id,
                    "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{self.model_id}",
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": top_k,
                        }
                    },
                },
            },
        )
        
        return self._format_response(response)
    
    def _format_response(self, response: dict) -> dict:
        """Parse API response into clean format."""
        output = response.get('output', {})
        
        sources = []
        for i, cit in enumerate(response.get('citations', [])):
            location = cit.get('location', {})
            sources.append({
                "id": i + 1,
                "title": location.get('document', {}).get('title', 'Unknown'),
                "uri": location.get('document', {}).get('uri', ''),
                "relevance_score": location.get('score', 0),
            })
        
        return {
            "answer": output.get('text', ''),
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        }

# Usage
rag = MinimalBedrockRAG(kb_id="kb-12345678")
result = rag.ask("How do I request AWS support?")

print(result['answer'])
print("\nSources:")
for s in result['sources']:
    print(f"  [{s['id']}] {s['title']} (score: {s['relevance_score']:.2f})")
```

## 10.2 RAG with LangChain + Bedrock

```python
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import OpenSearchVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import boto3

class LangChainBedrockRAG:
    """RAG pipeline using LangChain with Bedrock models."""
    
    def __init__(
        self,
        region: str = "us-east-1",
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        llm_model: str = "anthropic.claude-3-sonnet-4-20250514",
        opensearch_host: str = "search-xxx.us-east-1.aoss.amazonaws.com",
    ):
        self.region = region
        
        # Bedrock runtime client
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        
        # Embeddings
        self.embeddings = BedrockEmbeddings(
            model_id=embedding_model,
            region_name=region,
        )
        
        # LLM (via LangChain)
        from langchain.llms import Bedrock
        self.llm = Bedrock(
            model_id=llm_model,
            client=self.bedrock_runtime,
            model_kwargs={"temperature": 0.3, "max_tokens": 2048},
        )
        
        # Vector store
        self.vector_store = OpenSearchVectorStore(
            embedding=self.embeddings,
            index_name="rag-index",
            opensearch_url=f"https://{opensearch_host}",
        )
    
    def load_and_index_pdf(self, pdf_path: str):
        """Load PDF and index into vector store."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=lambda x: len(x.split()),
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        return len(chunks)
    
    def create_retriever(self, search_type: str = "similarity", k: int = 5):
        """Create a retriever for the vector store."""
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )
    
    def create_qa_chain(self):
        """Create a RetrievalQA chain with custom prompt."""
        prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, say you don't know. Be concise and cite your sources.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.create_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
    
    def ask(self, question: str) -> dict:
        """Ask a question and get an answer."""
        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [
                {"content": doc.page_content[:200], "metadata": doc.metadata}
                for doc in result.get("source_documents", [])
            ]
        }

# Usage
rag = LangChainBedrockRAG(
opensearch_host="search-xxx.us-east-1.aoss.amazonaws.com"
)

# Index documents
rag.load_and_index_pdf("./documents/policy.pdf")

# Ask questions
result = rag.ask("What is the password policy?")
print(result['answer'])
```

## 10.3 RAG with Custom Vector Store (Chroma + Bedrock)

```python
import chromadb
from chromadb.config import Settings
import boto3
import json

class ChromaBedrockRAG:
    """
    RAG using ChromaDB as local vector store + Bedrock for embeddings/LLM.
    Good for prototyping and small-scale deployments.
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Initialize Chroma
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def embed_titan(self, texts: list[str]) -> list[list[float]]:
        """Generate Titan embeddings for texts."""
        body = json.dumps({"inputText": texts if len(texts) > 1 else texts[0]})
        
        response = self.bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            body=body,
            accept='application/json',
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read().decode())
        
        if len(texts) == 1:
            return [result['embedding']]
        return result.get('embeddings', [result['embedding']])
    
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embed_titan([query])[0]
    
    def add_documents(self, texts: list[str], ids: list[str], metadatas: list[dict] = None):
        """Add documents to Chroma."""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Generate embeddings
        embeddings = self.embed_titan(texts)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve relevant documents."""
        query_emb = self.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        docs = []
        for i in range(len(results['ids'][0])):
            docs.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "distance": results['distances'][0][i],
                "similarity": 1 - results['distances'][0][i],
                "metadata": results['metadatas'][0][i],
            })
        
        return docs
    
    def generate_answer(
        self,
        question: str,
        context_docs: list[dict],
        model: str = "anthropic.claude-3-sonnet-4-20250514"
    ) -> str:
        """Generate answer using Bedrock LLM."""
        context = "\n\n".join([d['content'] for d in context_docs])
        
        prompt = f"""Use the following context to answer the question.
Context:
{context}

Question: {question}

Answer:"""
        
        body = json.dumps({
            "anthropic_version": "bedcore-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]
        })
        
        response = self.bedrock.invoke_model(
            modelId=model,
            body=body,
            accept='application/json',
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read().decode())
        return result['content'][0]['text']
    
    def ask(self, question: str, k: int = 5) -> dict:
        """Complete RAG: retrieve + generate."""
        docs = self.retrieve(question, k=k)
        answer = self.generate_answer(question, docs)
        
        return {
            "answer": answer,
            "sources": docs
        }

# Usage
rag = ChromaBedrockRAG(persist_dir="./my_rag_db")

# Add documents
rag.add_documents(
    texts=[
        "The password policy requires minimum 12 characters with mixed case, numbers, and symbols.",
        "AWS MFA must be enabled on the root account and all IAM users with console access.",
        "Security incidents must be reported within 24 hours to the security team."
    ],
    ids=["policy-1", "policy-2", "policy-3"],
    metadatas=[{"source": "security_policy", "version": "1.0"}] * 3
)

# Ask question
result = rag.ask("What are the password requirements?")
print(result['answer'])
```

## 10.4 Streaming RAG Responses

```python
import boto3
import json

class StreamingBedrockRAG:
    """
    RAG with streaming responses using Bedrock's streaming capability.
    Provides real-time token-by-token output for better UX.
    """
    
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
    
    def ask_streaming(self, question: str, top_k: int = 5):
        """
        Ask question with streaming response.
        Yields tokens as they arrive.
        """
        # First retrieve
        retrieve_response = self.client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": top_k}
            }
        )
        
        # Collect context from retrieved documents
        retrieval_results = retrieve_response.get('retrievalResults', [])
        context = "\n\n".join([
            r['content']['text'] for r in retrieval_results
        ])
        
        # Build prompt
        prompt = f"""Use the following context to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
        
        # Stream the LLM response
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]
        })
        
        # Use invoke_model_with_response_stream
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId='anthropic.claude-3-sonnet-4-20250514',
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        # Yield tokens as they arrive
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            
            if chunk['type'] == 'content_block_delta':
                if chunk['delta']['type'] == 'text_delta':
                    yield chunk['delta']['text']
    
    def ask_with_citations(self, question: str, top_k: int = 5) -> dict:
        """
        Non-streaming version that also returns citations.
        Use streaming for UX, this for structured responses.
        """
        response = self.client.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": self.kb_id,
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-4-20250514",
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {"numberOfResults": top_k}
                    },
                },
            },
        )
        
        citations = []
        for i, cit in enumerate(response.get('citations', [])):
            citations.append({
                "index": i + 1,
                "text": cit.get('generatedResponsePart', {}).get('textSegment', ''),
                "source": cit.get('location', {}).get('document', {}).get('title', 'Unknown'),
                "score": cit.get('location', {}).get('score', 0),
            })
        
        return {
            "answer": response['output']['text'],
            "citations": citations,
        }

# Usage with streaming
rag = StreamingBedrockRAG(kb_id="kb-12345")

print("Answer: ", end="", flush=True)
full_answer = ""
for token in rag.ask_streaming("How do I enable MFA?"):
    print(token, end="", flush=True)
    full_answer += token
print()  # Newline after streaming completes
```

## 10.5 Multi-Modal RAG on Bedrock

```python
import boto3
import json

class MultimodalBedrockRAG:
    """
    Multi-modal RAG: Query documents containing images, charts, and diagrams.
    Uses Titan Multimodal embeddings.
    """
    
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
    
    def embed_image_titan_multimodal(self, image_path: str) -> list[float]:
        """
        Generate embedding for an image using Titan Multimodal.
        Accepts image files (PNG, JPEG) or S3 URIs.
        """
        import base64
        
        # Read image and encode
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        body = json.dumps({
            "inputImage": image_data,
            "inputText": "Describe this image in detail for semantic search."
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId='amazon.titan-embed-image-v1:0',
            body=body,
            accept='application/json',
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read().decode())
        return result['embedding']
    
    def embed_text_and_search(self, text_query: str, k: int = 10) -> list[dict]:
        """
        Search for documents matching a text query.
        Works with both text-only and multi-modal documents.
        """
        # Embed the query text
        body = json.dumps({"inputText": text_query})
        response = self.bedrock_runtime.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            body=body,
            accept='application/json',
            contentType='application/json'
        )
        query_embedding = json.loads(response['body'].read().decode())['embedding']
        
        # Retrieve from KB (handles both text and image embeddings)
        retrieve_response = self.bedrock_agent.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": text_query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": k}
            }
        )
        
        return [
            {
                "content": r['content']['text'],
                "type": r.get('type', 'text'),  # 'text' or 'image'
                "source": r.get('location', {}).get('uri', ''),
                "score": r['score'],
            }
            for r in retrieve_response.get('retrievalResults', [])
        ]
    
    def ask_about_image(self, image_path: str, question: str) -> str:
        """
        Ask a question about a specific image.
        Useful for querying document figures, charts, diagrams.
        """
        # Embed the image
        image_embedding = self.embed_image_titan_multimodal(image_path)
        
        # For full image Q&A, use Claude with image input
        # This example shows semantic search for similar images
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(open(image_path, 'rb').read()).decode()
                            }
                        },
                        {"type": "text", "text": question}
                    ]
                }
            ]
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-sonnet-4-20250514',
            body=body,
            accept='application/json',
            contentType='application/json'
        )
        
        result = json.loads(response['body'].read().decode())
        return result['content'][0]['text']

# Usage
mm_rag = MultimodalBedrockRAG(kb_id="kb-multimodal")

# Search for documents containing charts about revenue
results = mm_rag.embed_text_and_search(
    "revenue growth chart quarterly analysis",
    k=10
)

# Ask about a specific chart image
answer = mm_rag.ask_about_image("./charts/q4-revenue.png", 
                                  "What does this chart show about Q4 revenue?")
```

## 10.6 Agent + Knowledge Base Integration

```python
from langchain.agents import Agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.tool import Tool
import boto3

class BedrockKBToolsAgent:
    """
    Agent that uses Bedrock KB as a tool.
    Agent can decide when to search the KB vs. use other tools.
    """
    
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.bedrock = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
        self.llm = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    def retrieve_from_kb(self, query: str) -> str:
        """Tool function: retrieve from KB and format results."""
        response = self.bedrock.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 5}
            }
        )
        
        results = response.get('retrievalResults', [])
        
        if not results:
            return "No relevant documents found."
        
        formatted = []
        for i, r in enumerate(results):
            formatted.append(
                f"[{i+1}] {r['content']['text']}\n"
                f"Source: {r.get('location', {}).get('title', 'Unknown')}\n"
            )
        
        return "\n---\n".join(formatted)
    
    def create_agent(self):
        """Create an agent with KB as a retrieval tool."""
        
        # Define KB retrieval tool
        kb_tool = Tool(
            name="CompanyKnowledgeBase",
            func=self.retrieve_from_kb,
            description="""Search the company knowledge base for information.
            Use this tool when the user asks about company policies, procedures,
            technical documentation, or any factual information that should be
            verified against company documents.
            
            Input: A search query string.
            Output: Relevant document excerpts with sources."""
        )
        
        # Additional tools could include: web search, calculator, etc.
        
        # Note: For full agent implementation, use LangChain agent framework
        # with the tool attached to a Claude-powered agent
        
        return kb_tool

# Usage
agent_tools = BedrockKBToolsAgent(kb_id="kb-12345")
kb_tool = agent_tools.create_agent()

# Example conversation flow:
# User: "What's our policy on remote work?"
# Agent: Decides to use KB tool
# Tool result: Returns policy document excerpts
# Agent: Synthesizes answer from retrieved documents
```

## 10.7 Guardrails with RAG

```python
import boto3

class BedrockRAGWithGuardrails:
    """
    RAG with Bedrock Guardrails integration for content safety.
    Applies safety filters to both retrieved content and generated responses.
    """
    
    def __init__(self, kb_id: str, guardrail_arn: str):
        self.kb_id = kb_id
        self.guardrail_arn = guardrail_arn
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
        self.bedrock = boto3.client('bedrock', region_name='us-east-1')
    
    def apply_guardrail_to_retrieval(self, retrieved_docs: list[dict]) -> list[dict]:
        """
        Apply guardrails to retrieved documents before including in context.
        Filter out documents that violate safety policies.
        """
        bedrock_guardrails = boto3.client('bedrock', region_name='us-east-1')
        
        safe_docs = []
        for doc in retrieved_docs:
            content = doc.get('content', '')
            
            # Analyze content against guardrail
            analysis = bedrock_guardrails.apply_guardrail(
                guardrailIdentifier=self.guardrail_arn,
                content=[{"text": content}]
            )
            
            # Check if content is safe
            if not analysis.get('action') == 'BLOCKED':
                safe_docs.append(doc)
            else:
                print(f"Filtered out document due to safety: {doc.get('id')}")
        
        return safe_docs
    
    def generate_with_guardrails(
        self,
        prompt: str,
        guardrail_arn: str,
        model: str = "anthropic.claude-3-sonnet-4-20250514"
    ) -> str:
        """
        Generate response with guardrails applied.
        Bedrock Guardrails can be applied at inference time.
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
            "guardrailIdentifier": guardrail_arn,
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        result = json.loads(response['body'].read().decode())
        
        # Check if response was modified by guardrail
        if 'guardrailAction' in result:
            print(f"Guardrail action: {result['guardrailAction']}")
        
        return result['content'][0]['text']
    
    def rag_with_guardrails(self, question: str, top_k: int = 5) -> dict:
        """
        Complete RAG pipeline with guardrails at both retrieval and generation.
        """
        # Step 1: Retrieve
        retrieve_response = self.bedrock_agent_runtime.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": top_k * 2}  # Over-fetch for filtering
            }
        )
        
        retrieved_docs = [
            {"content": r['content']['text'], "id": r.get('location', {}).get('uri')}
            for r in retrieve_response.get('retrievalResults', [])
        ]
        
        # Step 2: Apply guardrails to retrieved content
        safe_docs = self.apply_guardrail_to_retrieval(retrieved_docs)
        
        if not safe_docs:
            return {
                "answer": "I couldn't find any safe, relevant documents to answer your question.",
                "sources": [],
                "guardrail_filtered": True,
            }
        
        # Step 3: Assemble context
        context = "\n\n".join([d['content'] for d in safe_docs[:top_k]])
        
        # Step 4: Generate with guardrails
        full_prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.generate_with_guardrails(
            prompt=full_prompt,
            guardrail_arn=self.guardrail_arn
        )
        
        return {
            "answer": answer,
            "sources": safe_docs[:top_k],
            "guardrail_filtered": len(safe_docs) < len(retrieved_docs),
        }
```

---

# Part 11: Advanced RAG Patterns

## 11.1 Self-RAG (Self-Retrieval Augmented Generation)

Self-RAG uses the LLM itself to decide when to retrieve and how to use retrieved content. The model generates special tokens during generation to indicate:
- `[Retrieval]`: Need to retrieve for this claim
- `[No Retrieval]`: Can answer without retrieval
- `[Relevant]`: Retrieved document is relevant
- `[Partially Relevant]`: Retrieved document is partially relevant
- `[Not Relevant]`: Retrieved document is not relevant

```python
class SelfRAGPattern:
    """
    Self-RAG: LLM decides retrieval and evaluates relevance.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def is_retrieval_needed(self, question: str) -> bool:
        """
        Ask LLM if retrieval is needed.
        """
        prompt = f"""Does this question require consulting external documents to answer accurately?
Answer only YES or NO.

Question: {question}"""
        
        response = self.llm.invoke(prompt).strip().upper()
        return "YES" in response
    
    def evaluate_relevance(self, question: str, document: str) -> str:
        """
        Ask LLM to evaluate if document is relevant to question.
        """
        prompt = f"""Evaluate if the document helps answer the question.
Respond with exactly one word: RELEVANT, PARTIALLY_RELEVANT, or NOT_RELEVANT.

Question: {question}

Document: {document[:500]}..."""
        
        response = self.llm.invoke(prompt).strip().upper()
        if "NOT" in response:
            return "NOT_RELEVANT"
        elif "PARTIALLY" in response:
            return "PARTIALLY_RELEVANT"
        return "RELEVANT"
    
    def self_rag(self, question: str, retriever) -> str:
        """
        Self-RAG pipeline:
        1. Decide if retrieval needed
        2. If yes, retrieve and evaluate each doc
        3. Generate using only relevant docs
        """
        if not self.is_retrieval_needed(question):
            # Direct generation without retrieval
            return self.llm.invoke(question)
        
        # Retrieve candidate documents
        candidates = retriever.search(question, k=10)
        
        # Evaluate and filter
        relevant_docs = []
        for doc in candidates:
            relevance = self.evaluate_relevance(question, doc['content'])
            if relevance in ["RELEVANT", "PARTIALLY_RELEVANT"]:
                relevant_docs.append({
                    "doc": doc,
                    "relevance": relevance
                })
        
        if not relevant_docs:
            return "I don't have enough relevant information to answer this question."
        
        # Assemble context with relevance indicators
        context_parts = []
        for item in relevant_docs:
            relevance_marker = f"[{item['relevance']}]"
            context_parts.append(f"{relevance_marker}\n{item['doc']['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate with awareness of relevance
        prompt = f"""Use the following documents to answer the question.
Some documents may be marked as PARTIALLY_RELEVANT - use them only for supporting information.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.invoke(prompt)
```

## 11.2 Corrective RAG (CRAG)

CRAG adds a verification step between retrieval and generation. If retrieval quality is low, it triggers alternative actions (web search, rephrase query, etc.).

```python
class CorrectiveRAG:
    """
    CRAG: Verify retrieval quality and trigger corrections if needed.
    """
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def check_retrieval_quality(self, question: str, docs: list[dict]) -> dict:
        """
        Evaluate retrieval quality using LLM as judge.
        Returns: {"quality": "HIGH/MEDIUM/LOW", "reason": str}
        """
        if not docs:
            return {"quality": "LOW", "reason": "No documents retrieved"}
        
        top_score = docs[0].get('score', 0)
        
        if top_score < 0.5:
            return {"quality": "LOW", "reason": f"Top score {top_score:.2f} below threshold"}
        
        # Use LLM to verify semantic relevance
        prompt = f"""Evaluate if these documents actually answer the question.

Question: {question}

Top Document: {docs[0]['content'][:300]}...

Does this document contain information directly relevant to answering the question?
Respond with: YES, PARTIALLY, or NO."""
        
        response = self.llm.invoke(prompt).strip().upper()
        
        if "NO" in response:
            return {"quality": "LOW", "reason": "Documents don't answer question"}
        elif "PARTIALLY" in response:
            return {"quality": "MEDIUM", "reason": "Partially relevant"}
        return {"quality": "HIGH", "reason": "Good relevance"}
    
    def correct_and_generate(
        self,
        question: str,
        fallback_search=None
    ) -> str:
        """
        CRAG pipeline with corrective actions based on quality.
        """
        # Step 1: Initial retrieval
        docs = self.retriever.search(question, k=10)
        
        # Step 2: Evaluate quality
        quality = self.check_retrieval_quality(question, docs)
        
        if quality['quality'] == 'LOW':
            # Action 1: Query expansion / rephrasing
            expanded_queries = self._expand_query(question)
            
            all_docs = []
            for q in expanded_queries:
                all_docs.extend(self.retriever.search(q, k=5))
            
            # Deduplicate and re-rank
            docs = self._dedupe_and_rerank(all_docs)
            
            # Re-check quality
            quality = self.check_retrieval_quality(question, docs)
            
            if quality['quality'] == 'LOW' and fallback_search:
                # Action 2: Fallback to web search
                web_results = fallback_search(question)
                docs = self._merge_documents(docs, web_results)
        
        # Step 3: Generate from (possibly corrected) documents
        context = "\n\n".join([d['content'] for d in docs[:5]])
        
        return self.llm.invoke(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")
    
    def _expand_query(self, query: str) -> list[str]:
        """Generate alternative query formulations."""
        prompt = f"""Generate 3 different ways to ask this question.
Return only the alternative questions, one per line.

Original: {query}"""
        
        response = self.llm.invoke(prompt)
        queries = [query] + [q.strip() for q in response.split('\n') if q.strip()]
        return queries[:3]
    
    def _dedupe_and_rerank(self, docs: list[dict]) -> list[dict]:
        """Remove duplicates and re-rank."""
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            content_hash = hash(doc['content'][:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Re-sort by score
        return sorted(unique_docs, key=lambda x: x.get('score', 0), reverse=True)
```

## 11.3 Graph RAG (Knowledge Graph + Vectors)

Graph RAG combines vector search with knowledge graph traversal for richer context.

```python
class GraphRAG:
    """
    Graph RAG: Use knowledge graph for structured reasoning + vectors for retrieval.
    """
    
    def __init__(self, vector_store, kg_client):
        self.vector_store = vector_store
        self.kg = kg_client  # e.g., Neptune, Neo4j
    
    def extract_entities(self, question: str) -> list[dict]:
        """Extract named entities from question using LLM."""
        prompt = f"""Extract named entities (people, organizations, products, locations)
from this text. Return as JSON list of {{"name": str, "type": str}}.

Text: {question}"""
        
        response = self.llm.invoke(prompt)
        return json.loads(response)
    
    def query_knowledge_graph(
        self,
        entities: list[dict],
        relationship_type: str = None
    ) -> list[dict]:
        """
        Query knowledge graph for relationships between entities.
        Example: Find all relationships for 'AWS' entity.
        """
        results = []
        
        for entity in entities:
            # Cypher query to Neo4j or similar
            query = f"""
            MATCH (e {{name: '{entity['name']}'}})-[r]-(related)
            RETURN e, r, related
            LIMIT 20
            """
            kg_results = self.kg.run(query)
            results.extend(kg_results)
        
        return results
    
    def graph_rag(self, question: str) -> str:
        """
        Graph RAG: Combine KG context with vector retrieval.
        """
        # Step 1: Extract entities
        entities = self.extract_entities(question)
        
        # Step 2: Get graph context
        kg_context = ""
        if entities:
            kg_results = self.query_knowledge_graph(entities)
            kg_context = self._format_kg_results(kg_results)
        
        # Step 3: Vector retrieval
        vector_results = self.vector_store.search(question, k=5)
        vector_context = "\n\n".join([r['content'] for r in vector_results])
        
        # Step 4: Combine contexts
        combined_context = f"""Knowledge Graph Information:
{kg_context}

Document Information:
{vector_context}"""
        
        # Step 5: Generate with combined context
        prompt = f"""Use the following information to answer the question.
Information comes from a knowledge graph (structured relationships) and documents.

{combined_context}

Question: {question}

Answer:"""
        
        return self.llm.invoke(prompt)
```

## 11.4 Agentic RAG

Agentic RAG uses an agent to orchestrate multiple retrieval and reasoning steps.

```python
class AgenticRAG:
    """
    Agentic RAG: Use agent to decide retrieval strategy and orchestrate steps.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = {
            "vector_search": self._vector_search,
            "sql_query": self._sql_query,
            "calculate": self._calculate,
            "web_search": self._web_search,
        }
    
    def decide_action(self, question: str, history: list) -> dict:
        """
        LLM agent decides next action based on question and history.
        """
        prompt = f"""Given the user's question and conversation history, decide the next action.

Available tools:
- vector_search: Search company documents
- sql_query: Query structured database
- calculate: Perform calculations
- web_search: Search the internet

History: {history[-3:] if history else 'None'}

Question: {question}

Respond in JSON:
{{"action": "tool_name", "reasoning": "why", "query": "specific query for the tool"}}"""
        
        response = self.llm.invoke(prompt)
        return json.loads(response)
    
    def agentic_rag(self, question: str) -> str:
        """
        Execute agentic RAG with multiple reasoning steps.
        """
        history = []
        max_steps = 5
        
        for step in range(max_steps):
            # Decide next action
            decision = self.decide_action(question, history)
            action = decision['action']
            query = decision['query']
            
            if action == 'DONE':
                return decision.get('answer', "Here's my answer based on the information gathered.")
            
            # Execute action
            if action in self.tools:
                result = self.tools[action](query)
                history.append({"action": action, "query": query, "result": result})
            
            # Check if we have enough information
            if self._has_sufficient_info(history):
                break
        
        # Generate final answer from collected information
        return self._synthesize(question, history)
    
    def _has_sufficient_info(self, history: list) -> bool:
        """Check if we have enough information to answer."""
        if not history:
            return False
        
        # Simple heuristic: if we have 2+ retrievals with non-empty results
        retrievals = [h for h in history if 'search' in h.get('action', '')]
        return len(retrievals) >= 2
```

## 11.5 Ensemble RAG

Combine multiple retrievers with different strengths:

```python
class EnsembleRAG:
    """
    Ensemble RAG: Combine multiple retrieval strategies.
    """
    
    def __init__(self, retrievers: list, weights: list = None):
        """
        Args:
            retrievers: List of (name, retriever) tuples
            weights: Weight for each retriever (default: equal)
        """
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
    
    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        """
        Retrieve from all retrievers and fuse results.
        """
        all_results = {}
        
        for (name, retriever), weight in zip(self.retrievers, self.weights):
            results = retriever.search(query, k=k)
            
            for rank, result in enumerate(results):
                doc_id = result.get('doc_id', result.get('id', rank))
                
                # RRF score
                rrf_score = weight * (1 / (60 + rank + 1))
                
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'doc': result,
                        'score': rrf_score,
                        'sources': [name]
                    }
                else:
                    all_results[doc_id]['score'] += rrf_score
                    all_results[doc_id]['sources'].append(name)
        
        # Sort by fused score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [r['doc'] for r in sorted_results[:k]]
```

## 11.6 Hierarchical RAG

Two-stage retrieval: coarse (document-level) then fine (chunk-level):

```python
class HierarchicalRAG:
    """
    Hierarchical RAG: First retrieve documents, then relevant chunks within them.
    """
    
    def __init__(self, doc_vector_store, chunk_vector_store):
        self.doc_store = doc_vector_store      # Full documents
        self.chunk_store = chunk_vector_store  # Small chunks
    
    def retrieve(self, query: str, k_docs: int = 5, k_chunks: int = 10) -> list[dict]:
        """
        Two-stage: retrieve relevant documents, then chunks within them.
        """
        # Stage 1: Retrieve relevant documents
        docs = self.doc_store.search(query, k=k_docs)
        
        # Stage 2: For each doc, retrieve relevant chunks
        final_chunks = []
        for doc in docs:
            # Search for chunks belonging to this document
            chunks = self.chunk_store.search(
                query,
                k=k_chunks // k_docs,
                filter={"parent_doc_id": doc['doc_id']}
            )
            final_chunks.extend(chunks)
        
        # Re-rank all chunks
        return self._rerank_chunks(query, final_chunks)
```

## 11.7 Query Routing (Semantic Router)

Route queries to different retrieval strategies based on intent:

```python
class SemanticRouter:
    """
    Route queries to appropriate retrieval strategy based on semantics.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def route(self, query: str) -> str:
        """
        Determine which retrieval strategy to use.
        Returns: 'semantic', 'keyword', 'graph', 'hybrid', or 'general'
        """
        prompt = f"""Analyze this query and determine the best retrieval approach.

Approaches:
- semantic: For conceptual, meaning-based queries
- keyword: For specific terms, names, exact matches
- graph: For queries about relationships, comparisons
- hybrid: For queries needing both semantic and keyword
- general: For simple, direct questions

Query: {query}

Respond with only the approach name."""
        
        response = self.llm.invoke(prompt).strip().lower()
        
        valid = ['semantic', 'keyword', 'graph', 'hybrid', 'general']
        return response if response in valid else 'semantic'
    
    def retrieve(self, query: str) -> list[dict]:
        """
        Route and retrieve using appropriate strategy.
        """
        route = self.route(query)
        
        if route == 'semantic':
            return self.semantic_retriever.search(query)
        elif route == 'keyword':
            return self.bm25_retriever.search(query)
        elif route == 'graph':
            return self.graph_retriever.search(query)
        elif route == 'hybrid':
            return self.hybrid_retriever.search(query)
        else:
            return self.general_retriever.search(query)
```

## 11.8 Adaptive RAG

Adaptive RAG dynamically adjusts retrieval based on confidence:

```python
class AdaptiveRAG:
    """
    Adaptive RAG: Adjust strategy based on retrieval confidence.
    """
    
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.65
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def adaptive_retrieve(self, question: str) -> dict:
        """
        Adaptively retrieve based on confidence at each step.
        """
        # Step 1: Initial retrieval
        results = self.retriever.search(question, k=20)
        
        top_score = results[0].get('score', 0) if results else 0
        
        if top_score >= self.HIGH_CONFIDENCE:
            # High confidence: use top results directly
            return {"docs": results[:5], "strategy": "direct", "confidence": "high"}
        
        elif top_score >= self.MEDIUM_CONFIDENCE:
            # Medium: Expand query and re-rank
            expanded = self._expand_and_rerank(question, results)
            return {"docs": expanded, "strategy": "expanded", "confidence": "medium"}
        
        else:
            # Low: Try multiple strategies and fuse
            fused = self._multi_strategy_fusion(question)
            return {"docs": fused, "strategy": "fusion", "confidence": "low"}
    
    def _expand_and_rerank(self, question: str, results: list) -> list:
        """Query expansion + reranking for medium confidence."""
        # Expand query
        expanded_q = self.llm.invoke(f"Rephrase: {question}")
        
        # Retrieve with expanded query
        expanded = self.retriever.search(expanded_q, k=10)
        
        # Rerank combined results
        return self._rerank(question, results + expanded)
    
    def _multi_strategy_fusion(self, question: str) -> list:
        """Combine semantic, keyword, and expanded strategies."""
        semantic = self.retriever.search(question, k=10)
        keyword = self.bm25_retriever.search(question, k=10)
        
        # RRF fusion
        fused = self._rrf_fusion([semantic, keyword])
        return fused
```

---

---

# Part 12: Evaluation & Metrics

Evaluation is critical for production RAG systems. You need to measure both retrieval quality and generation quality to iteratively improve your system.

## 12.1 Retrieval Metrics

### 12.1.1 Precision@K

What fraction of the top-K retrieved documents are relevant?

```python
def precision_at_k(
    retrieved_docs: list[dict],
    relevant_doc_ids: set[str],
    k: int
) -> float:
    """
    Precision@K = (# of relevant docs in top K) / K
    
    Args:
        retrieved_docs: List of retrieved documents (ordered by rank)
        relevant_doc_ids: Set of document IDs that are truly relevant
        k: Number of top results to evaluate
    """
    if k <= 0:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    relevant_in_k = sum(1 for doc in retrieved_k if doc.get('doc_id') in relevant_doc_ids)
    
    return relevant_in_k / k
```

### 12.1.2 Recall@K

What fraction of ALL relevant documents were retrieved in top-K?

```python
def recall_at_k(
    retrieved_docs: list[dict],
    relevant_doc_ids: set[str],
    k: int
) -> float:
    """
    Recall@K = (# of relevant docs in top K) / (total # of relevant docs)
    
    Note: This metric is only meaningful if you know ALL relevant docs
    for the query, which requires a labeled dataset.
    """
    total_relevant = len(relevant_doc_ids)
    if total_relevant == 0:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    relevant_in_k = sum(1 for doc in retrieved_k if doc.get('doc_id') in relevant_doc_ids)
    
    return relevant_in_k / total_relevant
```

### 12.1.3 Mean Reciprocal Rank (MRR)

The reciprocal of the rank of the first relevant document. High MRR = relevant docs appear early.

```python
def mean_reciprocal_rank(
    queries: list[str],
    retrieval_results: dict[str, list[dict]],
    relevant_docs: dict[str, set[str]],
) -> float:
    """
    MRR = average of (1 / rank_of_first_relevant_doc) across all queries
    
    If no relevant doc found, contribution is 0.
    """
    reciprocal_ranks = []
    
    for query in queries:
        docs = retrieval_results[query]
        relevant = relevant_docs[query]
        
        for rank, doc in enumerate(docs, 1):
            if doc.get('doc_id') in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
```

### 12.1.4 NDCG@K (Normalized Discounted Cumulative Gain)

Measures ranking quality, accounting for the position of relevant documents (relevant docs at top matter more).

```python
def dcg_at_k(scores: list[float], k: int) -> float:
    """DCG = sum of (relevance_score / log2(rank+1))"""
    dcg = 0.0
    for i, score in enumerate(scores[:k]):
        dcg += score / np.log2(i + 2)  # i+2 because rank starts at 1 and log2(1) = 0
    return dcg

def ndcg_at_k(
    retrieved_docs: list[dict],
    relevant_doc_ids: set[str],
    k: int
) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    
    where IDCG is the ideal DCG (relevant docs perfectly ranked at top)
    """
    # Relevance scores: 1 if relevant, 0 otherwise
    relevance_scores = [
        1.0 if doc.get('doc_id') in relevant_doc_ids else 0.0
        for doc in retrieved_docs[:k]
    ]
    
    # Actual DCG
    dcg = dcg_at_k(relevance_scores, k)
    
    # Ideal DCG (all relevant docs at top, with 1.0 relevance)
    num_relevant = min(len(relevant_doc_ids), k)
    ideal_relevance = [1.0] * num_relevant + [0.0] * (k - num_relevant)
    idcg = dcg_at_k(ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0.0
```

## 12.2 Generation Metrics

### 12.2.1 Faithfulness

Does the generated answer match the retrieved context? Measures hallucination.

```python
def faithfulness(
    answer: str,
    context: str,
    llm,
) -> float:
    """
    Measure faithfulness using LLM as judge.
    
    Faithfulness = (# of claims verifiable from context) / (total # of claims)
    """
    # Extract claims from answer
    claims_prompt = f"""Extract all factual claims from this answer.
Return each claim on a separate line. If a sentence has no facts, skip it.

Answer: {answer}"""
    
    claims = llm.invoke(claims_prompt).strip().split('\n')
    claims = [c.strip() for c in claims if c.strip()]
    
    if not claims:
        return 1.0  # No claims to verify = vacuously faithful
    
    # Verify each claim against context
    verifiable_count = 0
    for claim in claims:
        verify_prompt = f"""Can this claim be verified from the context?
Answer YES or NO only.

Claim: {claim}

Context: {context}"""
        
        response = llm.invoke(verify_prompt).strip().upper()
        if "YES" in response:
            verifiable_count += 1
    
    return verifiable_count / len(claims)
```

### 12.2.2 Answer Relevance

Does the answer actually address the question?

```python
def answer_relevance(
    question: str,
    answer: str,
    llm,
) -> float:
    """
    Measure answer relevance using LLM to generate alternative questions.
    
    Relevance = average similarity between original question and LLM-generated alternatives.
    Higher similarity = answer better addresses the question.
    """
    # Generate alternative questions the answer would also address
    alt_q_prompt = f"""Generate 3 questions that this answer would also address.
Return one question per line.

Answer: {answer}"""
    
    alt_questions = llm.invoke(alt_q_prompt).strip().split('\n')
    alt_questions = [q.strip() for q in alt_questions if q.strip()]
    
    if not alt_questions:
        return 0.0
    
    # Embed both original question and alternatives
    original_emb = embed_text(question)
    alt_embs = [embed_text(q) for q in alt_questions]
    
    # Calculate average similarity
    similarities = [cosine_sim(original_emb, emb) for emb in alt_embs]
    avg_similarity = sum(similarities) / len(similarities)
    
    return avg_similarity
```

### 12.2.3 Context Precision and Recall

**Context Precision:** What fraction of the context is actually relevant to answering?

**Context Recall:** What fraction of the information needed to answer is in the context?

```python
def context_precision(
    retrieved_contexts: list[str],
    question: str,
    ground_truth: str,
    llm,
) -> float:
    """
    Context Precision = How much of the retrieved context is relevant?
    """
    relevant_contexts = 0
    
    for context in retrieved_contexts:
        # Does this context contribute to answering?
        check_prompt = f"""Does this context contain information useful for answering the question?
Answer YES or NO.

Question: {question}

Context: {context}"""
        
        response = llm.invoke(check_prompt).strip().upper()
        if "YES" in response:
            relevant_contexts += 1
    
    return relevant_contexts / len(retrieved_contexts) if retrieved_contexts else 0.0

def context_recall(
    retrieved_contexts: list[str],
    question: str,
    ground_truth: str,
    llm,
) -> float:
    """
    Context Recall = Does the retrieved context contain the information in ground truth?
    """
    context_combined = "\n\n".join(retrieved_contexts)
    
    recall_prompt = f"""Compare the retrieved context to the ground truth answer.
Estimate what fraction of the ground truth information (0.0 to 1.0) is present in the context.

Ground Truth: {ground_truth}

Retrieved Context: {context_combined}

Respond with just a number between 0.0 and 1.0."""
    
    response = llm.invoke(recall_prompt).strip()
    try:
        return float(response)
    except:
        return 0.0
```

## 12.3 RAGAS Framework

RAGAS (RAG Assessment) is a framework for automated RAG evaluation using LLM-as-judge:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics import FactualCorrectness

class RAGASEvaluator:
    """
    Evaluate RAG pipeline using RAGAS metrics.
    
    Requires:
    - questions: List of test questions
    - answers: Ground truth answers
    - contexts: Retrieved contexts for each question
    - responses: Generated answers
    """
    
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            FactualCorrectness(),
        ]
    
    def evaluate(
        self,
        questions: list[str],
        ground_truths: list[str],
        contexts: list[list[str]],
        responses: list[str],
    ) -> dict:
        """
        Run RAGAS evaluation.
        
        Returns dict of metric_name -> score pairs.
        """
        # Create dataset
        from datasets import Dataset
        
        data = {
            "question": questions,
            "ground_truth": ground_truths,
            "answer": responses,
            "contexts": contexts,
        }
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        result = evaluate(dataset, metrics=self.metrics)
        
        return result
```

## 12.4 ARES Framework

ARES (Automated RAG Evaluation System) from Stanford uses lightweightLM judges for faster, cheaper evaluation:

```python
class ARESEvaluation:
    """
    ARES evaluation using lightweight judges for efficiency.
    
    Key difference from RAGAS: Uses smaller, specialized models
    for judgments instead of large frontier models.
    """
    
    def __init__(self):
        self.judge_model = "lightweight-judge-model"  # e.g., Llama-3-8B
    
    def evaluate_single(
        self,
        question: str,
        response: str,
        context: list[str],
        ground_truth: str = None,
    ) -> dict:
        """
        Evaluate a single RAG response using ARES criteria.
        """
        return {
            "faithfulness": self._judge_faithfulness(response, context),
            "answer_relevance": self._judge_relevance(question, response),
            "context_relevance": self._judge_context_relevance(question, context),
            "factual_correctness": self._judge_factual(response, ground_truth) if ground_truth else None,
        }
```

## 12.5 Custom Evaluation Pipeline

```python
class RAGEvaluator:
    """
    Complete evaluation pipeline for RAG systems.
    """
    
    def __init__(self, retriever, llm, embedder):
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
    
    def evaluate_retrieval(
        self,
        test_queries: list[dict],  # [{question, relevant_docs}]
    ) -> dict:
        """
        Evaluate retrieval quality on labeled test set.
        
        Args:
            test_queries: List of {question, relevant_docs: set} dicts
        """
        metrics = {"precision_at_5": [], "recall_at_5": [], "mrr": [], "ndcg_at_5": []}
        
        for test in test_queries:
            query = test['question']
            relevant = test['relevant_docs']
            
            # Retrieve
            query_emb = self.embedder.embed_query(query)
            results = self.retriever.search(query_emb, k=20)
            
            # Calculate metrics
            metrics["precision_at_5"].append(precision_at_k(results, relevant, 5))
            metrics["recall_at_5"].append(recall_at_k(results, relevant, 5))
            
            # MRR
            mrr_score = 0.0
            for rank, doc in enumerate(results, 1):
                if doc.get('doc_id') in relevant:
                    mrr_score = 1.0 / rank
                    break
            metrics["mrr"].append(mrr_score)
            
            # NDCG
            metrics["ndcg_at_5"].append(ndcg_at_k(results, relevant, 5))
        
        # Average across all queries
        return {
            f"avg_{k}": sum(v) / len(v) if v else 0.0
            for k, v in metrics.items()
        }
    
    def evaluate_generation(
        self,
        test_queries: list[dict],  # [{question, answer, context}]
    ) -> dict:
        """
        Evaluate generation quality.
        """
        metrics = {"faithfulness": [], "answer_relevance": [], "factual_correctness": []}
        
        for test in test_queries:
            answer = test['answer']
            context = test['context']
            question = test['question']
            
            # Faithfulness
            metrics["faithfulness"].append(faithfulness(answer, context, self.llm))
            
            # Answer relevance
            metrics["answer_relevance"].append(answer_relevance(question, answer, self.llm))
            
            # Factual correctness
            if 'ground_truth' in test:
                metrics["factual_correctness"].append(
                    self._factual_correctness(answer, test['ground_truth'])
                )
        
        return {f"avg_{k}": sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}
    
    def full_evaluation(self, test_dataset: list[dict]) -> dict:
        """
        Run complete evaluation (retrieval + generation).
        """
        retrieval_results = []
        generation_results = []
        
        for test in test_dataset:
            query = test['question']
            query_emb = self.embedder.embed_query(query)
            
            # Retrieval
            retrieved = self.retriever.search(query_emb, k=10)
            contexts = [doc['content'] for doc in retrieved]
            
            # Generate
            context_combined = "\n\n".join(contexts)
            prompt = f"Context: {context_combined}\n\nQuestion: {query}\n\nAnswer:"
            response = self.llm.invoke(prompt)
            
            retrieval_results.append({
                "question": query,
                "retrieved": retrieved,
                "relevant_docs": test.get('relevant_docs', set()),
            })
            
            generation_results.append({
                "question": query,
                "answer": response,
                "context": contexts,
                "ground_truth": test.get('ground_truth'),
            })
        
        return {
            "retrieval": self.evaluate_retrieval(retrieval_results),
            "generation": self.evaluate_generation(generation_results),
        }
```

## 12.6 Human Evaluation Protocols

Automated metrics complement but don't replace human evaluation:

```python
HUMAN_EVALUATION_TEMPLATE = """
## RAG Response Evaluation

**Question:** {question}

**Retrieved Context:**
{contexts}

**Generated Response:**
{response}

**Rate each dimension (1-5):**

1. **Accuracy** (Does the answer match the context?): ___/5
   - 1: Contains factual errors
   - 3: Mostly accurate with minor issues
   - 5: Completely accurate

2. **Relevance** (Does it answer the question?): ___/5
   - 1: Doesn't address the question
   - 3: Partially answers
   - 5: Fully addresses the question

3. **Completeness** (Is it thorough enough?): ___/5
   - 1: Missing key information
   - 3: Covers main points but incomplete
   - 5: Comprehensive

4. **Clarity** (Is it well-written?): ___/5
   - 1: Confusing, poorly structured
   - 3: Understandable but not polished
   - 5: Clear and well-organized

5. **Citation Quality** (Are sources properly attributed?): ___/5
   - 1: No citations or incorrect
   - 3: Some citations
   - 5: Clear, accurate citations

**Comments:**
{comments}
"""

def create_human_eval_batch(
    questions: list[str],
    responses: list[str],
    contexts: list[list[str]],
    batch_size: int = 20,
) -> list[str]:
    """
    Create human evaluation forms for a batch of RAG outputs.
    """
    forms = []
    for q, r, c in zip(questions, responses, contexts):
        form = HUMAN_EVALUATION_TEMPLATE.format(
            question=q,
            response=r,
            contexts="\n---\n".join([f"[{i+1}] {ctx[:500]}..." for i, ctx in enumerate(c)]),
            comments="_ " * 30,
        )
        forms.append(form)
    
    return forms
```

## 12.7 Continuous Evaluation in Production

```python
class ProductionRAGEvaluator:
    """
    Continuous evaluation for production RAG systems.
    Logs metrics and alerts on degradation.
    """
    
    def __init__(self, metrics_client):
        self.metrics = metrics_client  # CloudWatch, Datadog, etc.
    
    def log_retrieval_latency(self, latency_ms: float):
        self.metrics.put_metric("RAG/RetrievalLatency", latency_ms, "Milliseconds")
    
    def log_generation_latency(self, latency_ms: float):
        self.metrics.put_metric("RAG/GenerationLatency", latency_ms, "Milliseconds")
    
    def log_retrieval_quality(
        self,
        query: str,
        retrieved_docs: list[dict],
        user_feedback: int,  # 1-5 stars
    ):
        """
        Log user feedback correlated with retrieval results.
        Low feedback + low retrieval scores = retrieval problem.
        Low feedback + high retrieval scores = generation problem.
        """
        avg_score = sum(d.get('score', 0) for d in retrieved_docs[:3]) / 3
        
        self.metrics.put_metric_data([
            {"MetricName": "RAG/UserFeedback", "Value": user_feedback},
            {"MetricName": "RAG/Top3RetrievalScore", "Value": avg_score},
        ])
        
        # Alert if quality degrades
        if user_feedback < 3 and avg_score > 0.8:
            self._alert("Low feedback despite high retrieval scores - investigate generation")
        elif user_feedback < 3 and avg_score < 0.5:
            self._alert("Low feedback with low retrieval scores - investigate retrieval")
```

---

# Part 13: Production Operations

## 13.1 Monitoring RAG Pipelines

### 13.1.1 Key Metrics to Track

| Metric | What It Measures | Alert Threshold |
|--------|-----------------|----------------|
| Retrieval Latency | Vector search speed | > 100ms p95 |
| Generation Latency | LLM response time | > 5s p95 |
| End-to-End Latency | Total RAG time | > 10s p95 |
| Retrieval Score (avg) | Quality of top results | < 0.6 avg |
| Empty Retrieval Rate | % queries with no results | > 5% |
| Token Usage | Cost tracking | Unexpected spikes |
| Error Rate | System health | > 1% |
| User Feedback | Satisfaction | < 3.5 stars |

### 13.1.2 CloudWatch Dashboard Configuration

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create dashboard
dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "title": "RAG Latency (p95)",
                "metrics": [
                    ["RAG", "RetrievalLatency", "p95"],
                    [".", "GenerationLatency", "p95"],
                    [".", "TotalLatency", "p95"],
                ],
                "period": 60,
                "stat": "Percentile",
                "region": "us-east-1",
                "yAxis": {"left": {"min": 0, "max": 10000}},
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "Retrieval Quality",
                "metrics": [
                    ["RAG", "AvgRetrievalScore", "avg"],
                    [".", "EmptyRetrievalRate", "avg"],
                ],
                "period": 300,
                "stat": "Average",
                "region": "us-east-1",
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "User Satisfaction",
                "metrics": [
                    ["RAG", "UserFeedback", "avg"],
                    [".", "FeedbackCount", "sum"],
                ],
                "period": 3600,
                "stat": "Average",
                "region": "us-east-1",
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "Costs",
                "metrics": [
                    ["AWS/Bedrock", "InferenceTokens", "Sum", "Model", "claude-3-sonnet"],
                    [".", "EmbeddingTokens", "Sum"],
                ],
                "period": 3600,
                "stat": "Sum",
                "region": "us-east-1",
            }
        },
    ]
}

cloudwatch.put_dashboard(
    DashboardName="RAG-Production-Monitoring",
    DashboardBody=json.dumps(dashboard_body)
)
```

### 13.1.3 Structured Logging

```python
import logging
import json
from datetime import datetime

class RAGLogger:
    """Structured logging for RAG operations."""
    
    def __init__(self, log_group: str = "/aws/bedrock/rag"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # CloudWatch handler
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.handler)
    
    def log_retrieval(
        self,
        query_id: str,
        query: str,
        num_results: int,
        top_score: float,
        latency_ms: float,
        filters: dict = None,
    ):
        self.logger.info(json.dumps({
            "event": "retrieval",
            "query_id": query_id,
            "query_hash": hash(query) % 1000000,  # For deduplication, not storage
            "num_results": num_results,
            "top_score": round(top_score, 4),
            "latency_ms": latency_ms,
            "filters_applied": filters,
            "timestamp": datetime.utcnow().isoformat(),
        }))
    
    def log_generation(
        self,
        query_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str,
    ):
        self.logger.info(json.dumps({
            "event": "generation",
            "query_id": query_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
        }))
    
    def log_error(
        self,
        query_id: str,
        error_type: str,
        error_message: str,
        stage: str,  # "retrieval", "generation", "indexing"
    ):
        self.logger.error(json.dumps({
            "event": "error",
            "query_id": query_id,
            "error_type": error_type,
            "error_message": error_message[:500],  # Truncate
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat(),
        }))
```

## 13.2 Latency Optimization

### 13.2.1 Latency Breakdown

```
Total Latency = Retrieval Latency + Generation Latency

Retrieval Latency = Embed Query + Vector Search + Network
Generation Latency = Prompt Tokens Transmission + LLM Inference + Response Transmission

Typical breakdown (p95):
- Embed query: 50-100ms
- Vector search (10K vectors): 20-50ms  
- Vector search (1M vectors): 50-150ms
- LLM TTFT (time to first token): 200-500ms
- LLM per-token: 10-30ms × output tokens
- Total generation (500 tokens): 3-8 seconds
```

### 13.2.2 Optimization Strategies

```python
class LatencyOptimizer:
    """
    Strategies to reduce RAG latency.
    """
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def optimize_retrieval(self, query: str) -> list[dict]:
        """
        Optimized retrieval pipeline.
        """
        # 1. Pre-compute and cache query embedding if this is a common query
        cache_key = hash(query) % 1000000
        cached_emb = self._get_cached_embedding(cache_key)
        
        if cached_emb:
            query_emb = cached_emb
        else:
            query_emb = self.embedder.embed_query(query)
            self._cache_embedding(cache_key, query_emb)
        
        # 2. Use async for parallel operations where possible
        # 3. Use smaller embedding model if latency critical
        # 4. Reduce k if fewer results acceptable
        
        return self.vector_store.search(query_emb, k=5)
    
    def optimize_generation(self, prompt: str, model: str = "claude-3-haiku-20240307") -> str:
        """
        Generation optimizations:
        - Use smaller/faster model for simple queries
        - Reduce max_tokens if responses are short
        - Enable streaming for perceived latency
        """
        # Route to appropriate model based on query complexity
        complexity = self._estimate_complexity(prompt)
        
        if complexity == "simple":
            model = "anthropic.claude-3-haiku-20240307"  # Faster, cheaper
        elif complexity == "moderate":
            model = "anthropic.claude-3-sonnet-4-20250514"
        # else use full model
        
        # Stream for better UX (first token sooner)
        return self._stream_generate(prompt, model)
    
    def _estimate_complexity(self, prompt: str) -> str:
        """Simple heuristic for query complexity."""
        words = len(prompt.split())
        if words < 10:
            return "simple"
        elif words < 30:
            return "moderate"
        return "complex"
```

## 13.3 Cost Tracking and Optimization

```python
class CostTracker:
    """
    Track and optimize RAG costs.
    """
    
    def __init__(self):
        self.daily_costs = {}
        self.cost_by_query_type = {}
    
    def record_embedding_cost(self, num_tokens: int, model: str = "titan-v2"):
        cost_per_1k = 0.0002  # Titan v2
        cost = (num_tokens / 1000) * cost_per_1k
        self._add_cost("embeddings", cost)
    
    def record_generation_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ):
        pricing = {
            "claude-3-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }
        p = pricing.get(model, pricing["claude-3-sonnet-4-20250514"])
        
        cost = (input_tokens / 1000) * p["input"]
        cost += (output_tokens / 1000) * p["output"]
        
        self._add_cost(f"generation_{model}", cost)
    
    def _add_cost(self, category: str, cost: float):
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_costs:
            self.daily_costs[today] = {}
        if category not in self.daily_costs[today]:
            self.daily_costs[today][category] = 0
        self.daily_costs[today][category] += cost
    
    def get_monthly_forecast(self) -> float:
        """Estimate next month's cost based on current trends."""
        if len(self.daily_costs) < 7:
            return None  # Not enough data
        
        avg_daily = sum(
            sum(costs.values()) for costs in self.daily_costs.values()
        ) / len(self.daily_costs)
        
        return avg_daily * 30

# Cost optimization: reduce embedding dimensions
def optimize_embedding_storage():
    """
    Titan v2: reduce dimensions from 1536 to 1024 saves ~33% storage
    with ~2% recall degradation.
    """
    pass
```

## 13.4 Scaling Ingestion

```python
class ScalableIngestion:
    """
    Scale document ingestion for large document corpus.
    """
    
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
    
    async def ingest_large_corpus(
        self,
        documents: list[Document],
        batch_size: int = 100,
        max_workers: int = 10,
    ):
        """
        Asynchronous parallel ingestion.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        
        # Chunk all documents first (CPU-bound)
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
        
        # Embed in parallel batches
        all_embeddings = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                texts = [c.page_content for c in batch]
                
                # Submit embedding batch
                future = loop.run_in_executor(
                    executor,
                    self.embedder.embed_documents,
                    texts
                )
                
                # Process completed futures
                while True:
                    try:
                        embeddings = await asyncio.wait_for(future, timeout=60)
                        all_embeddings.extend(embeddings)
                        break
                    except asyncio.TimeoutError:
                        print(f"Batch {i//batch_size} timed out, retrying...")
                        future = loop.run_in_executor(
                            executor,
                            self.embedder.embed_documents,
                            texts
                        )
        
        # Bulk index to vector store
        self.vector_store.add_texts(
            texts=[c.page_content for c in all_chunks],
            embeddings=all_embeddings,
            metadatas=[c.metadata for c in all_chunks],
        )
        
        return len(all_chunks)
```

## 13.5 Handling Updates to Source Documents

See Section 5.5 for IncrementalIndexer implementation.

## 13.6 A/B Testing Retrievers

```python
class ABTestRetrievers:
    """
    A/B test different retrieval configurations.
    """
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(
        self,
        name: str,
        variant_a: dict,  # Config for variant A
        variant_b: dict,  # Config for variant B
        traffic_split: float = 0.5,  # % to variant A
    ):
        """
        Create A/B test for retrieval configs.
        """
        self.experiments[name] = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "traffic_split": traffic_split,
            "results_a": [],
            "results_b": [],
        }
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Deterministically assign user to variant."""
        exp = self.experiments[experiment_name]
        # Hash user_id for consistent assignment
        hash_val = hash(user_id) % 100
        if hash_val < exp["traffic_split"] * 100:
            return "a"
        return "b"
    
    def record_result(
        self,
        experiment_name: str,
        variant: str,
        query: str,
        user_feedback: float,
        retrieval_scores: list[float],
    ):
        """Record outcome of an experiment."""
        exp = self.experiments[experiment_name]
        result = {
            "query": query,
            "user_feedback": user_feedback,
            "avg_retrieval_score": sum(retrieval_scores) / len(retrieval_scores),
            "timestamp": datetime.now().isoformat(),
        }
        
        if variant == "a":
            exp["results_a"].append(result)
        else:
            exp["results_b"].append(result)
    
    def analyze_experiment(self, experiment_name: str) -> dict:
        """Statistical analysis of A/B test results."""
        exp = self.experiments[experiment_name]
        
        results_a = exp["results_a"]
        results_b = exp["results_b"]
        
        if len(results_a) < 30 or len(results_b) < 30:
            return {"status": "insufficient_data"}
        
        # Calculate means
        mean_a = sum(r["user_feedback"] for r in results_a) / len(results_a)
        mean_b = sum(r["user_feedback"] for r in results_b) / len(results_b)
        
        # Simple z-test for significance
        # (In production, use proper statistical library)
        
        return {
            "variant_a": {
                "n": len(results_a),
                "mean_feedback": mean_a,
                "mean_retrieval_score": sum(r["avg_retrieval_score"] for r in results_a) / len(results_a),
            },
            "variant_b": {
                "n": len(results_b),
                "mean_feedback": mean_b,
                "mean_retrieval_score": sum(r["avg_retrieval_score"] for r in results_b) / len(results_b),
            },
            "winner": "a" if mean_a > mean_b else "b",
            "improvement_pct": abs(mean_a - mean_b) / max(mean_a, mean_b) * 100,
        }
```

---

# Part 14: Security, Privacy, Compliance

## 14.1 Data Residency in RAG

### 14.1.1 AWS Regional Considerations

```
When using Bedrock KB, data flows through:
1. S3 bucket (your region) → Bedrock ingestion (your region)
2. Bedrock → OpenSearch Serverless collection (your region)
3. Query → LLM inference (your region's Bedrock endpoint)

Key: All data stays within the specified AWS region.
```

### 14.1.2 Multi-Region Data Residency

```python
def create_compliant_kb(
    region: str,
    s3_bucket: str,
    data_classification: str,  # "public", "internal", "confidential", "restricted"
):
    """
    Create region-compliant knowledge base based on data classification.
    """
    bedrock = boto3.client('bedrock', region_name=region)
    
    # For restricted data, ensure:
    if data_classification == "restricted":
        # Use customer-managed keys (CMK)
        kms_client = boto3.client('kms', region_name=region)
        key_arn = kms_client.create_key([
            {"AttributeName": "aws:resourcedomainusage", "AttributeValue": "bedrock-kb"}
        ])['KeyMetadata']['ARN']
        
        # Enable encryption at rest with CMK
        encryption_config = {"kmsKeyArn": key_arn}
    else:
        encryption_config = {}  # Use default AWS-managed encryption
    
    kb = bedrock.create_knowledge_base(
        name=f"compliant-kb-{data_classification}",
        # ... other config
    )
    
    return kb
```

## 14.2 PII Handling in Documents

### 14.2.1 Detection and Redaction

```python
import re

class PIIHandler:
    """
    Detect and handle PII in documents before embedding.
    """
    
    def __init__(self):
        # Patterns for common PII (use AWS Comprehend in production)
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
    
    def detect_pii(self, text: str) -> list[dict]:
        """
        Detect PII in text. Returns list of (pii_type, span, value).
        """
        findings = []
        
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                findings.append({
                    "type": pii_type,
                    "start": match.start(),
                    "end": match.end(),
                    "value": match.group(),
                })
        
        return findings
    
    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Replace all PII with redaction marker.
        """
        redacted = text
        
        # Sort by start position descending to not mess up offsets
        findings = sorted(self.detect_pii(text), key=lambda x: x['start'], reverse=True)
        
        for finding in findings:
            redacted = (
                redacted[:finding['start']]
                + f"[{finding['type'].upper()}]{replacement}[/{finding['type'].upper()}]"
                + redacted[finding['end']:]
            )
        
        return redacted
    
    def extract_and_store_separately(
        self,
        text: str,
        store: dict,
    ) -> str:
        """
        Extract PII, store securely, replace with placeholder.
        """
        findings = self.detect_pii(text)
        redacted = text
        
        # Sort descending for safe replacement
        findings = sorted(findings, key=lambda x: x['start'], reverse=True)
        
        for i, finding in enumerate(findings):
            placeholder = f"__PII_{i}__"
            redacted = (
                redacted[:finding['start']]
                + placeholder
                + redacted[finding['end']:]
            )
            
            # Store securely (encrypted, access-controlled)
            store[placeholder] = {
                "type": finding['type'],
                "value": finding['value'],  # Encrypt in production
            }
        
        return redacted

# AWS Comprehend for production PII detection
def detect_pii_comprehend(text: str, region: str = "us-east-1") -> list[dict]:
    """
    Use AWS Comprehend for accurate PII detection.
    """
    comprehend = boto3.client('comprehend', region_name=region)
    
    response = comprehend.detect_pii_entities(
        Text=text,
        LanguageCode='en'
    )
    
    return [
        {"type": r['Type'], "start": r['BeginOffset'], "end": r['EndOffset']}
        for r in response['Entities']
    ]
```

## 14.3 Encryption at Rest and In Transit

### 14.3.1 Encryption Configuration

```python
# S3 bucket encryption (Server-side encryption)
s3 = boto3.client('s3')

s3.put_bucket_encryption(
    Bucket='my-kb-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [
            {
                'ApplyServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'AES256'  # or 'aws:kms' for KMS
                },
                'BucketKeyEnabled': True
            }
        ]
    }
)

# OpenSearch Serverless encryption
# Automatic with AWS-managed keys or customer-managed CMK
# Via KMS key policy restriction
```

### 14.3.2 TLS Configuration

```
All Bedrock API calls use TLS 1.2+ automatically.
OpenSearch Serverless requires HTTPS.

VPC endpoint configuration for private connectivity:
- Bedrock runtime API: bedrock-runtime.us-east-1.vpce.amazonaws.com
- Bedrock agent API: bedrock-agent.us-east-1.vpce.amazonaws.com
```

## 14.4 Access Control on Vector Stores

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::123456789:role/kb-access-role"},
      "Action": [
        "aoss:APIAccessAll"
      ],
      "Resource": "arn:aws:aoss:us-east-1:123456789:collection/xxx",
      "Condition": {
        "Bool": {"aws:SecureTransport": "true"}
      }
    },
    {
      "Effect": "Deny",
      "Principal": "*",
      "Action": "aoss:*",
      "Resource": "arn:aws:aoss:us-east-1:123456789:collection/xxx",
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": ["10.0.0.0/8"]}  # Restrict to corporate network
      }
    }
  ]
}
```

## 14.5 Audit Logging

```python
import boto3

class RAGAuditLogger:
    """
    Comprehensive audit logging for compliance.
    """
    
    def __init__(self, audit_trail_bucket: str):
        self.cloudtrail = boto3.client('cloudtrail')
        self.s3 = boto3.client('s3')
        self.bucket = audit_trail_bucket
    
    def log_query(
        self,
        user_id: str,
        query: str,
        response: str,
        sources_accessed: list[str],
        kb_id: str,
    ):
        """
        Log each RAG query for audit trail.
        """
        audit_entry = {
            "event_type": "RAG_QUERY",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "query_hash": hash(query),  # Don't store query text
            "query_length": len(query),
            "kb_id": kb_id,
            "sources_accessed": sources_accessed,
            "response_length": len(response),
            "event_version": "1.0",
        }
        
        # Write to S3 with user-specific prefix for fast retrieval
        key = f"audit/{user_id}/{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4()}.json"
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(audit_entry),
            ServerSideEncryption='aws:kms',
        )
    
    def log_data_access(
        self,
        user_id: str,
        document_id: str,
        action: str,  # "view", "download", "print"
        kb_id: str,
    ):
        """Log document-level access."""
        access_entry = {
            "event_type": "DOCUMENT_ACCESS",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "document_id": document_id,
            "action": action,
            "kb_id": kb_id,
        }
        
        # Write to CloudWatch Logs for real-time monitoring
        cloudwatch = boto3.client('logs')
        cloudwatch.put_log_events(
            logGroupName='/aws/bedrock/kb/audit',
            logStreamName=user_id,
            logEvents=[{
                'timestamp': int(datetime.utcnow().timestamp() * 1000),
                'message': json.dumps(access_entry),
            }]
        )
```

## 14.6 Compliance Considerations

### 14.6.1 GDPR Compliance Checklist

```python
GDPR_CHECKLIST = {
    "data_minimization": {
        "description": "Only collect data necessary for the purpose",
        "rag_applicable": "Only embed/document chunks necessary for the use case",
        "checked": False,
    },
    "purpose_limitation": {
        "description": "Data used only for stated purpose",
        "rag_applicable": "RAG outputs used only for the Q&A purpose, not reused for analytics",
        "checked": False,
    },
    "storage_limitation": {
        "description": "Data not kept longer than necessary",
        "rag_applicable": "Implement document retention policies, delete old embeddings",
        "checked": False,
    },
    "right_to_deletion": {
        "description": "Ability to delete personal data",
        "rag_applicable": "Ability to remove documents from KB and delete associated embeddings",
        "checked": False,
    },
    "right_to_access": {
        "description": "Users can request their data",
        "rag_applicable": "Document what data is in the KB for a given user query",
        "checked": False,
    },
}
```

### 14.6.2 HIPAA Considerations

```python
HIPAA_REQUIREMENTS = {
    "encryption": "All PHI must be encrypted at rest (AES-256) and in transit (TLS 1.2+)",
    "access_control": "PHI access must be restricted and logged",
    "audit_trails": "All PHI access must be logged with user identity",
    "baa_required": "AWS BAA covers Bedrock when configured correctly",
}

def configure_hipaa_compliant_kb(kb_id: str):
    """
    Configure Bedrock KB for HIPAA compliance.
    """
    # Use CMK for encryption
    # Enable all audit logging
    # Restrict access to authorized roles only
    # Implement data retention policies
    pass
```

## 14.7 Amazon Bedrock Guardrails Integration

See Section 10.7 for guardrails implementation with RAG.

---

---

# Part 15: Troubleshooting Guide

## 15.1 Common Failure Modes and Solutions

### 15.1.1 Retrieval Quality Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Empty results | No matching docs in KB | Verify data synced, check filters |
| Irrelevant results | Chunking too large/small | Adjust chunk size, see Section 3.8 |
| Misses obvious matches | Embedding model mismatch | Try different embed model |
| Returns same doc repeatedly | MMR not enabled | Enable diversity in retrieval |
| Score always ~0.5 | Index not built properly | Rebuild vector index |

### 15.1.2 Generation Quality Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Hallucinates facts | Poor retrieval + no citation check | Enable faithfulness verification |
| Ignores retrieved context | Prompt doesn't emphasize context | Strengthen system prompt |
| Too verbose | No max_tokens or instruction | Add token limit to prompt |
| Wrong format | Conflicting instructions | Simplify prompt structure |
| Repetitive | LLM temperature too low | Increase temperature slightly |

## 15.2 Debugging Retrieval Quality Issues

```python
def debug_retrieval(
    question: str,
    retriever,
    embedder,
    top_k: int = 10,
) -> dict:
    """
    Debug retrieval by examining top results in detail.
    """
    # Embed query
    query_emb = embedder.embed_query(question)
    
    # Retrieve candidates
    results = retriever.search(query_emb, k=top_k)
    
    # Analyze each result
    analysis = []
    for i, result in enumerate(results):
        # Compute similarity manually
        doc_emb = result.get('embedding')
        if doc_emb:
            similarity = cosine_sim(query_emb, doc_emb)
        else:
            similarity = result.get('score', 'unknown')
        
        analysis.append({
            "rank": i + 1,
            "doc_id": result.get('doc_id'),
            "score": similarity,
            "content_preview": result.get('content', '')[:200],
            "metadata": result.get('metadata', {}),
        })
    
    # Also check what the query embedding looks like
    # and what terms it emphasizes
    
    return {
        "query": question,
        "query_embedding_stats": {
            "dimension": len(query_emb),
            "norm": np.linalg.norm(query_emb),
            "top_values": sorted(enumerate(query_emb), key=lambda x: abs(x[1]), reverse=True)[:10],
        },
        "results": analysis,
    }

def check_embedding_space(
    texts: list[str],
    embedder,
) -> dict:
    """
    Verify embeddings are in expected range (normalized, reasonable distribution).
    """
    embeddings = embedder.embed_documents(texts)
    
    stats = {
        "num_embeddings": len(embeddings),
        "dimension": len(embeddings[0]) if embeddings else 0,
        "norms": [np.linalg.norm(e) for e in embeddings],
        "min_norm": min(np.linalg.norm(e) for e in embeddings) if embeddings else 0,
        "max_norm": max(np.linalg.norm(e) for e in embeddings) if embeddings else 0,
    }
    
    # Check if normalized (norm ≈ 1.0 for Titan v2)
    if 0.99 < stats['min_norm'] < 1.01 and 0.99 < stats['max_norm'] < 1.01:
        stats['normalized'] = True
    else:
        stats['normalized'] = False
        print("WARNING: Embeddings not normalized as expected")
    
    return stats
```

## 15.3 Debugging Generation Quality Issues

```python
def debug_generation(
    question: str,
    context: str,
    llm,
) -> dict:
    """
    Debug generation by checking intermediate steps.
    """
    # Step 1: Check if context contains answer
    check_prompt = f"""Does this context contain the answer to the question?
Answer YES or NO, and explain briefly.

Question: {question}

Context: {context[:1000]}..."""
    
    context_contains_answer = llm.invoke(check_prompt)
    
    # Step 2: Check if answer matches context
    generate_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
    
    answer = llm.invoke(generate_prompt)
    
    # Step 3: Verify answer against context
    verify_prompt = f"""Does this answer use only information from the context?
Answer YES or specify any information not in the context.

Context: {context}

Answer: {answer}"""
    
    faithfulness = llm.invoke(verify_prompt)
    
    return {
        "context_contains_answer": context_contains_answer,
        "generated_answer": answer,
        "faithfulness_check": faithfulness,
    }
```

## 15.4 Performance Debugging

```python
import time

def profile_retrieval_latency(
    queries: list[str],
    embedder,
    vector_store,
) -> dict:
    """
    Profile retrieval latency breakdown.
    """
    latencies = {
        "embedding": [],
        "search": [],
        "total": [],
    }
    
    for query in queries:
        # Embed
        start = time.time()
        emb = embedder.embed_query(query)
        embed_lat = (time.time() - start) * 1000
        
        # Search
        start = time.time()
        results = vector_store.search(emb, k=10)
        search_lat = (time.time() - start) * 1000
        
        latencies["embedding"].append(embed_lat)
        latencies["search"].append(search_lat)
        latencies["total"].append(embed_lat + search_lat)
    
    return {
        "embedding_p50": np.percentile(latencies["embedding"], 50),
        "embedding_p95": np.percentile(latencies["embedding"], 95),
        "search_p50": np.percentile(latencies["search"], 50),
        "search_p95": np.percentile(latencies["search"], 95),
        "total_p50": np.percentile(latencies["total"], 50),
        "total_p95": np.percentile(latencies["total"], 95),
    }
```

## 15.5 Infrastructure Debugging (OpenSearch, IAM, Sync Failures)

### 15.5.1 OpenSearch Debugging

```python
def diagnose_opensearch_index(index_name: str, host: str):
    """
    Check OpenSearch index health and configuration.
    """
    import requests
    
    # Get index stats
    stats_url = f"https://{host}/{index_name}/_stats"
    stats = requests.get(stats_url, auth=auth).json()
    
    # Get index mapping
    mapping_url = f"https://{host}/{index_name}/_mapping"
    mapping = requests.get(mapping_url, auth=auth).json()
    
    # Check HNSW parameters
    index_settings_url = f"https://{host}/{index_name}/_settings"
    settings = requests.get(index_settings_url, auth=auth).json()
    
    return {
        "doc_count": stats['_all']['primaries']['docs']['count'],
        "index_size_bytes": stats['_all']['primaries']['store']['size_in_bytes'],
        "vector_field_config": mapping[index_name]['mappings'].get('properties', {}).get('vector'),
        "knn_settings": settings[index_name]['settings']['index']['knn'],
    }
```

### 15.5.2 Common Sync Failures

```python
SYNC_FAILURE_DIAGNOSIS = {
    "ACCESS_DENIED": {
        "cause": "IAM role lacks permissions to S3 bucket or OpenSearch",
        "fix": "Attach s3:GetObject policy to KB role, verify OpenSearch access policy",
    },
    "RESOURCE_NOT_FOUND": {
        "cause": "S3 bucket doesn't exist or KB doesn't exist",
        "fix": "Verify bucket ARN, ensure KB is created before sync",
    },
    "INVALID_DOCUMENT_FORMAT": {
        "cause": "Unsupported file format or corrupted file",
        "fix": "Convert to PDF, HTML, MD, TXT, or DOCX",
    },
    "DOCUMENT_SIZE_EXCEEDED": {
        "cause": "Document > 50MB limit",
        "fix": "Split document into smaller files",
    },
    "VECTOR_DIMENSION_MISMATCH": {
        "cause": "Embedding model changed after initial sync",
        "fix": "Delete and recreate KB, or re-embed all documents",
    },
}
```

---

# Part 16: Interview Q&A — RAG & Bedrock KB

## 50+ Interview Questions with Detailed Answers

### 16.1 Fundamentals (1-15)

**Q1: What problem does RAG solve that fine-tuning cannot?**

**A:** RAG solves three problems fine-tuning cannot adequately address:

1. **Knowledge currency**: Fine-tuned models have frozen knowledge at training time. RAG retrieves current information from a document store, making it always up-to-date without retraining.

2. **Citation and attribution**: RAG can provide exact document sources for each answer claim. Fine-tuning bakes knowledge into weights with no traceable source.

3. **Private data isolation**: RAG keeps documents in your infrastructure. Fine-tuning on private data bakes it into the model weights, risking exposure.

Fine-tuning is better for: adapting model personality/writing style, teaching reasoning patterns, reducing prompt size for domain-specific interactions.

---

**Q2: Explain the RAG pipeline at a high level.**

**A:** RAG has two pipelines:

**Ingestion**: Documents → Parse & Extract → Chunk → Embed → Index → Vector Store

**Query**: Question → Embed Query → Vector Search → Retrieve Chunks → Assemble Context → LLM Generate → Response with Citations

The key insight is separation of concerns: embeddings handle semantic search; the LLM handles generation from the retrieved context.

---

**Q3: What is a vector embedding? Why does it enable semantic search?**

**A:** An embedding is a dense numerical vector representation of text, where similar meanings map to nearby points in high-dimensional space (typically 256-4096 dimensions).

The magic: "How do I reset my password?" and "I forgot my login credentials" are semantically similar even without shared words. Both embed to nearby vectors, enabling retrieval by semantic meaning rather than keyword matching.

---

**Q4: What is cosine similarity and why is it used for vector search?**

**A:** Cosine similarity measures the angle between two vectors: `cos(θ) = (A·B) / (||A|| × ||B||)`.

Range: -1 to +1 (for normalized vectors: 0 to +1). Higher = more similar.

For L2-normalized embeddings (like Titan v2), dot product equals cosine similarity, making computation O(d) rather than requiring division.

---

**Q5: What is HNSW and why is it used for vector search?**

**A:** Hierarchical Navigable Small World (HNSW) is an ANN algorithm using multi-layer graphs:
- Top layers: sparse, enable long-range jumps
- Bottom layers: dense, enable precise local search
- Search: Start at top, greedily traverse to nearest neighbor, descend

Achieves 97-99% recall with ~30 distance computations vs 1M for brute force. Trade-off: O(n log n) build time, 1.2-1.5x memory overhead.

---

**Q6: What is the difference between dense and sparse retrieval?**

**A:**

**Dense (vector)**: Neural embeddings capture semantic meaning. "Dog" and "puppy" are similar even without shared terms. Computed with transformer models.

**Sparse (BM25/TF-IDF)**: Traditional IR based on term frequency. Exact keyword matches score high. Fast but miss synonyms and semantic relationships.

**Hybrid**: Combines both using RRF (Reciprocal Rank Fusion) for better coverage than either alone.

---

**Q7: Why is chunking important in RAG?**

**A:** Chunking determines retrieval granularity:
- Too large: Includes irrelevant context, wastes context window
- Too small: Loses surrounding context for understanding
- Optimal: Self-contained chunks that answer specific questions

Chunk size affects embedding quality, retrieval precision, and generation context. No universal answer — depends on document structure and use case.

---

**Q8: What is Maximum Marginal Relevance (MMR) and why use it?**

**A:** MMR balances relevance and diversity in retrieval:

`MMR = λ × relevance(query, doc) - (1-λ) × max_similarity(doc, selected_docs)`

Without MMR, top-k retrieval returns redundant results (same source, same viewpoint). MMR ensures selected documents are both relevant AND diverse, improving coverage.

---

**Q9: What is HyDE (Hypothetical Document Embeddings)?**

**A:** HyDE generates a hypothetical "ideal" document answering the query, then embeds that document for retrieval:

1. LLM generates a plausible document answering the query
2. Embed the hypothetical document (not the query)
3. Retrieve using the hypothetical's embedding

Intuition: The hypothetical shares embedding space with real documents, capturing the "answer style" which may better match document language than the query.

---

**Q10: What is a cross-encoder vs bi-encoder for re-ranking?**

**A:**

**Bi-encoder**: Encodes query and document separately. Fast (pre-computed doc embeddings) but doesn't capture query-document interaction.

**Cross-encoder**: Jointly encodes [query, document] pair. More accurate but too slow for initial retrieval over millions of docs.

**Typical pattern**: Bi-encoder retrieves top-100 candidates, cross-encoder re-ranks top-10 for final context.

---

**Q11: What is the "lost in the middle" problem?**

**A:** LLMs have attention bottlenecks: they better recall information at the beginning (primacy) and end (recency) of context, with degraded recall for information in the middle.

Solutions: Place important information at context boundaries; use summarization for long contexts; implement hierarchical retrieval that returns key paragraphs.

---

**Q12: What is Reciprocal Rank Fusion (RRF)?**

**A:** RRF combines rankings from multiple retrieval methods:

`RRF_score(doc) = Σ weight × (1 / (k + rank))`

k=60 is standard. Documents appearing at rank 1 in both retrievers get `2/(60+1)` = 0.0328; rank 1 in one, rank 5 in another gets `1/(60+1) + 1/(60+5)` = 0.0215.

Simple, no training needed, handles different score distributions across retrievers.

---

**Q13: What is Semantic Chunking vs Fixed-Size Chunking?**

**A:**

**Fixed-size**: Splits text by token/character count regardless of meaning. Fast but may split mid-sentence, mid-paragraph.

**Semantic**: Uses embeddings to detect topic boundaries. Sentences with high similarity stay together; split when similarity drops. Better coherence but slower and requires threshold tuning.

**Hybrid/Recursive**: Tries semantic splits first (paragraphs, sentences), falls back to smaller splits if chunk too large. Best of both for most cases.

---

**Q14: How does Parent Document Retriever work?**

**A:** Two-level retrieval architecture:

1. **Small chunks** (e.g., 500 tokens): Indexed for precise retrieval
2. **Large parent chunks** (e.g., 2000 tokens): Stored in document store

At query time: retrieve matching child chunks → fetch their parent documents → return parent context for generation.

Provides granular retrieval with full document context for the LLM.

---

**Q15: What is Self-RAG?**

**A:** Self-RAG uses the LLM to self-reflect on retrieval and generation:

1. LLM generates `[Retrieval]` or `[No Retrieval]` tokens to decide if retrieval needed
2. For retrieved documents, LLM generates `[Relevant]`, `[Partially Relevant]`, or `[Not Relevant]`
3. LLM uses these reflections to weight which documents to use

End-to-end differentiable: model learns when to retrieve and how to use retrieved content.

---

### 16.2 Bedrock Knowledge Bases (16-30)

**Q16: What is Amazon Bedrock Knowledge Bases?**

**A:** A fully managed RAG service on AWS that handles:
- Document ingestion from S3, Confluence, SharePoint, Salesforce, web crawlers
- Automatic chunking with multiple strategies
- Embedding generation (Titan or Cohere)
- Vector storage (OpenSearch Serverless, Aurora pgvector, Pinecone, Redis, MongoDB)
- Retrieval via `RetrieveAndGenerate` or `Retrieve` APIs
- Sync job management

Built on existing AWS services, integrates with IAM, CloudWatch, VPC.

---

**Q17: How does Bedrock KB differ from building RAG from scratch?**

**A:**

| Aspect | DIY RAG | Bedrock KB |
|--------|---------|------------|
| Ingestion pipeline | Build & maintain | Managed |
| Embedding infrastructure | You manage | One API call |
| Vector store ops | You manage | Managed (for OpenSearch Serverless) |
| Sync management | Custom solution | Built-in scheduled/event-driven |
| Cost model | All infrastructure | Pay per query + storage |
| Customization | Full control | Limited but sufficient |
| Integration | Manual | Native AWS integration |

---

**Q18: What chunking strategies does Bedrock KB support?**

**A:** Four strategies:

1. **NONE**: Entire document as one chunk (only for <8K token docs)
2. **FIXED_SIZE**: Split by token count with overlap. Simple but ignores semantics.
3. **HIERARCHICAL** (default): Small child chunks (500 tokens) + larger parent chunks (2000 tokens). Retrieves children, returns parents.
4. **SEMANTIC**: Split at semantic boundaries using embedding similarity. Best coherence but slowest.

---

**Q19: What data sources does Bedrock KB support?**

**A:**
- **S3**: PDF, HTML, Markdown, TXT, DOCX, CSV, JSON, PPTX
- **Confluence**: Wiki pages and attachments
- **SharePoint Online**: Document libraries
- **Salesforce**: Knowledge articles, Cases
- **Web Crawler**: Seed URLs or sitemap.xml

For S3, cross-account access supported with bucket policy. For Confluence/SharePoint/Salesforce, credentials stored in AWS Secrets Manager.

---

**Q20: What is the `RetrieveAndGenerate` API?**

**A:** Single API that performs complete RAG:

```python
bedrock_runtime.retrieve_and_generate(
    input={"text": question},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": kb_id,
            "modelArn": model_arn,
            "retrievalConfiguration": {...},
            "generationConfiguration": {...},
        }
    }
)
```

Returns answer + citations linking response text to source documents with confidence scores.

---

**Q21: When would you use `Retrieve` API vs `RetrieveAndGenerate`?**

**A:**

**Use `Retrieve` when:**
- Building custom RAG pipeline with your own prompt engineering
- Need to post-process retrieved documents (filter, rerank, compress)
- Want to combine KB results with other data sources
- Need to log/monitor retrieval separately from generation

**Use `RetrieveAndGenerate` when:**
- Quick implementation, minimal customization needed
- Standard RAG with default prompt is acceptable
- Don't need intermediate retrieval results

---

**Q22: What vector stores does Bedrock KB support?**

**A:**

| Store | Type | Best For |
|-------|------|----------|
| OpenSearch Serverless | AWS SaaS | Default, AWS-native |
| Aurora PostgreSQL (pgvector) | AWS RDS | Existing Aurora, ACID needs |
| Pinecone | SaaS | Global, fully managed |
| Redis | Self/SaaS | Ultra-low latency |
| MongoDB Atlas | SaaS | MongoDB users |

OpenSearch Serverless is default and recommended for AWS-native deployments.

---

**Q23: How do you optimize Bedrock KB costs?**

**A:**

1. **Reduce embedding dimensions**: Titan v2 supports 1536→1024→768→512 (lower = cheaper storage, ~2% recall loss per reduction step)
2. **Tune retrieval k**: Default 10 may be excessive; 3-5 for simple Q&A
3. **Use smaller models for simple queries**: Claude Haiku vs Sonnet
4. **Implement query caching**: Cache frequent queries
5. **Schedule syncs off-peak**: OpenSearch Serverless usage-based pricing
6. **Monitor per-query cost**: Track token usage per query type

---

**Q24: How does cross-region knowledge base access work?**

**A:** Bedrock KB operates within a single region. For cross-region:

1. **Separate KBs per region**: Each region has its own KB reading from regional S3 replica
2. **Route via API Gateway**: API Gateway in each region calls regional Bedrock
3. **Data residency**: Each region's KB reads only from that region's data sources

Cross-region read replicas for the vector store (e.g., Aurora Global Database) can provide DR but not lower latency.

---

**Q25: What IAM permissions does Bedrock KB require?**

**A:** The KB service role needs:
- `s3:GetObject` + `s3:ListBucket` on the source S3 bucket(s)
- `aoss:APIAccessAll` on the vector store collection
- `secretsmanager:GetSecretValue` for data source credentials (Confluence, SharePoint, etc.)
- `bedrock:*` on knowledge base resources

Best practice: Create dedicated role with minimal permissions scoped to specific resources.

---

**Q26: How do you handle sync job failures?**

**A:**

1. **Check job status** via `get_ingestion_job` for failure reason
2. **Common causes**:
   - `ACCESS_DENIED`: IAM permissions issue
   - `RESOURCE_NOT_FOUND`: Bucket/collection doesn't exist
   - `INVALID_DOCUMENT_FORMAT`: Unsupported file type
   - `DOCUMENT_SIZE_EXCEEDED`: File > 50MB
3. **CloudWatch logs**: `/aws/bedrock/knowledge-base` log group
4. **Partial failures**: Check `statistics.documentsFailed` — can be retried for failed docs
5. **Event-driven sync**: Use S3 EventBridge trigger for automatic reprocessing

---

**Q27: What are Bedrock KB's quotas and how do you request increases?**

**A:** Key quotas:
- 50 KBs per region (can increase)
- 10 data sources per KB
- 1M documents per KB
- 50MB max document size
- 100 max retrieval results (k)
- 5 concurrent sync jobs per KB
- 100 TPS for RetrieveAndGenerate

Request via AWS Support → Service Limit Increase → Bedrock → specific limit type.

---

**Q28: How does metadata filtering work in Bedrock KB retrieval?**

**A:** Filter in `vectorSearchConfiguration`:

```python
"filter": {
    "andAll": [
        {"key": "document_type", "value": "policy", "operator": "EQ"},
        {"key": "version", "value": "2024-01", "operator": "GTE"},
    ]
}
```

Supported operators: `EQ`, `NE`, `GTE`, `LTE`, `CONTAINS`, `IN`

Filters are post-filtered on OpenSearch results (may return fewer than `numberOfResults`).

---

**Q29: What is Titan Embeddings v2's Matryoshka dimension reduction?**

**A:** Titan v2 supports truncating embeddings to lower dimensions post-generation without retraining:

- Full: 1536 dimensions
- Can reduce to: 1024, 768, 512, 256

Storage savings: 1536 → 512 = 67% reduction with ~5-10% recall degradation.

Importantly: dimensions are ordered by importance (Matryoshka property), so truncation keeps most critical information.

---

**Q30: How do you implement VPC connectivity for Bedrock KB?**

**A:** Create VPC Interface Endpoints for Bedrock APIs:

```python
ec2.create_vpc_endpoint(
    VpcId='vpc-xxx',
    ServiceName='com.amazonaws.us-east-1.bedrock-agent-runtime',
    VpcEndpointType='Interface',
    SubnetIds=['subnet-xxx'],
    SecurityGroupIds=['sg-xxx'],
    PrivateDnsEnabled=True,
)
```

Required for:
- `bedrock-agent-runtime` (for RetrieveAndGenerate)
- `bedrock-runtime` (for direct LLM calls)

Security group needs port 443 outbound.

---

### 16.3 Advanced Patterns & Architecture (31-45)

**Q31: Design a RAG system that handles 10M documents.**

**A:** Key architectural decisions:

1. **Sharding strategy**: Partition by document type or tenant. Each shard has own vector store index.

2. **Hierarchical retrieval**: 
   - Coarse: Which shard/index to search
   - Fine: Vector search within shard

3. **Async ingestion pipeline**:
   - Message queue (SQS) for document processing jobs
   - Multiple workers processing chunks in parallel
   - Embedding API batch calls

4. **Caching layer**:
   - Query embedding cache (Redis/Memcached)
   - Frequently asked Q&A cache

5. **Monitoring**:
   - Per-shard metrics
   - Alert on retrieval degradation

6. **Cost optimization**:
   - Lower embedding dimensions (512 instead of 1536)
   - Delete old document versions
   - Archive inactive data to S3

---

**Q32: How would you implement multi-modal RAG?**

**A:**

1. **Ingestion**: 
   - Extract text and images from documents (PDF, DOCX)
   - Use multi-modal embedding model (Titan Multimodal) to embed images
   - Store both in vector store with type metadata

2. **Retrieval**:
   - Text query → embed with text model → search
   - Image query → embed with multimodal model → search
   - Both retrievals combined with RRF

3. **Generation**:
   - Pass retrieved text + image URIs to LLM with vision capability
   - LLM can "see" retrieved images and reason about them

4. **Handling images in context**:
   - Use Claude 3+ or GPT-4V for vision
   - Base64 encode images or use S3 URIs
   - Monitor token budget (images are expensive)

---

**Q33: What is Graph RAG and when would you use it?**

**A:** Graph RAG combines vector search with knowledge graph traversal:

1. **Extract entities** from documents (LLM-based NER)
2. **Build knowledge graph** of entity relationships
3. **Query**: Retrieve relevant graph subgraphs + vector results
4. **Generate**: Use combined graph + document context

**Best for:**
- Questions about relationships ("How is X related to Y?")
- Multi-hop reasoning ("What caused X to affect Y?")
- Corporate knowledge with rich entity relationships

**Not for:** Simple Q&A with direct answers from single documents.

---

**Q34: How do you evaluate RAG quality in production?**

**A:** Multi-layer approach:

1. **Offline evaluation**:
   - Labeled test set with ground truth answers
   - Metrics: Precision@K, Recall@K, NDCG@K, MRR (retrieval)
   - RAGAS/ARES metrics (generation): Faithfulness, Answer Relevance, Context Precision/Recall

2. **Continuous monitoring**:
   - Retrieval score distribution (alert if avg drops)
   - User feedback (thumbs up/down, star ratings)
   - Error rates and latency

3. **A/B testing**:
   - Test different chunk sizes, retrievers, prompts
   - Statistical significance testing on user feedback

4. **Human evaluation**:
   - Periodic sample review by experts
   - Annotate for faithfulness and relevance

---

**Q35: Explain RAGAS metrics.**

**A:** RAGAS (RAG Assessment) uses LLMs as judges:

1. **Faithfulness**: Does generated answer match retrieved context? (LLM verifies each claim)

2. **Answer Relevancy**: Does the answer address the question? (Generate alternative questions the answer would fit)

3. **Context Precision**: What fraction of retrieved context is relevant? (LLM judges each context)

4. **Context Recall**: Does retrieved context contain ground truth? (Compare to labeled answers)

5. **Factual Correctness**: Are facts in answer correct vs ground truth?

All metrics use LLM-as-judge with specific prompting templates.

---

**Q36: What is Corrective RAG (CRAG)?**

**A:** CRAG adds verification and self-correction:

1. **After retrieval**: Evaluate if retrieved docs actually answer the question
2. **If low quality**: Trigger query expansion (rephrase, add synonyms) and re-retrieve
3. **If still low**: Fall back to web search or escalate to human
4. **If high quality**: Proceed to generation

```
Query → Retrieve → Evaluate Quality
                       ↓
         Low? → Expand Query → Re-retrieve
                       ↓
         Still Low? → Web Search / Escalate
                       ↓
         High? → Generate
```

---

**Q37: How do you handle contradictory retrieved information?**

**A:** Strategies:

1. **Detect contradictions**: LLM checks retrieved docs for conflicts
2. **Present both perspectives**: "According to Doc A: X. According to Doc B: Y."
3. **Prioritize recency**: If dates differ, newer document may supersede
4. **Prioritize authority**: Official policy over unofficial notes
5. **Flag uncertainty**: "The documents appear to conflict on X"

Prompt engineering: Include explicit instruction to flag conflicts and present multiple views with sources.

---

**Q38: What is agentic RAG?**

**A:** Agentic RAG uses an LLM agent to orchestrate retrieval:

1. Agent receives question
2. Agent decides: retrieve from KB? Query SQL? Web search? Calculate?
3. Agent executes action(s) and evaluates results
4. Agent may iterate with refined queries
5. Agent synthesizes final answer

**Use cases:**
- Questions requiring multiple data sources
- Complex questions needing step-by-step reasoning
- Dynamic tool selection based on query type

Example: "Compare our Q4 revenue to competitors" → Agent queries internal DB + web search + calculates.

---

**Q39: How would you implement query routing in RAG?**

**A:** Route queries to appropriate retrieval strategy:

```python
routes = {
    "semantic_conceptual": semantic_retriever,
    "exact_keyword": bm25_retriever,
    "graph_relationship": graph_retriever,
    "hybrid": hybrid_retriever,
}

def route(query: str) -> str:
    # LLM classifies query type
    prompt = f"Classify: {query}\nTypes: semantic, keyword, graph, hybrid"
    route_type = llm.invoke(prompt)
    return route_type

def retrieve(query: str):
    route_type = route(query)
    return routes[route_type].search(query)
```

Classify by: presence of exact terms, question type (who/what/how), complexity, domain.

---

**Q40: What is adaptive RAG?**

**A:** Adaptive RAG dynamically adjusts strategy based on confidence:

1. **Query complexity estimation**: Simple factual → direct LLM, complex → full RAG
2. **Retrieval confidence**: High similarity scores → use directly, low → expand/retry
3. **Self-correction**: Generate answer, verify against context, regenerate if needed

Key difference from static RAG: Model decides on-the-fly how much retrieval to use.

---

**Q41: How do you handle document versioning in RAG?**

**A:** Options:

1. **Metadata versioning**: Store version number on each chunk. At query, filter to latest version or include all for comparison.

2. **Timestamp-based**: Store `indexed_at` on chunks. Query can specify time range.

3. **Separate indices**: Create new index per version. Switch alias atomically when ready.

4. **Soft delete**: Mark old chunks as deleted, filter them out. Re-index periodically.

Recommendation: Metadata approach is simplest. For strict compliance, separate indices with aliases.

---

**Q42: What is hierarchical RAG?**

**A:** Two-stage retrieval:

1. **Document-level retrieval**: Fast coarse search to find relevant documents
2. **Chunk-level retrieval**: Within those documents, find specific relevant chunks

```
Query → Fast document retrieval (coarse) → Identify top 5 documents
                                      ↓
           Fine chunk retrieval within those 5 documents
                                      ↓
                           Final chunk ranking
```

Balances recall (document level) with precision (chunk level).

---

**Q43: How do you implement hybrid search with Bedrock KB?**

**A:** Bedrock KB's `RetrieveAndGenerate` supports `overrideSearchType: "HYBRID"`:

```python
"retrievalConfiguration": {
    "vectorSearchConfiguration": {
        "overrideSearchType": "HYBRID",
        "numberOfResults": 20,
    }
}
```

HYBRID combines semantic (vector) and keyword (BM25) search internally.

For custom hybrid (with different weighting), use `Retrieve` API with your own fusion logic.

---

**Q44: What are the security considerations for RAG?**

**A:**

1. **Data residency**: All data stays in your region. Use regional S3 buckets and OpenSearch collections.

2. **Access control**: IAM roles for KB access, resource-based policies on S3/OpenSearch.

3. **Encryption**: At rest (AES-256 via KMS) and in transit (TLS 1.2+).

4. **PII handling**: Detect and redact before embedding, or use PII-safe chunk metadata.

5. **Audit logging**: Log all queries and document access to immutable store.

6. **Guardrails**: Apply Bedrock Guardrails to filter harmful content in both queries and responses.

7. **VPC**: Private connectivity via VPC endpoints.

---

**Q45: How do you optimize RAG for low latency?**

**A:**

1. **Embedding cache**: Cache query embeddings for repeated/frequent queries (Redis)
2. **Result cache**: Cache LLM responses for identical queries
3. **Precompute**: Batch-embed all documents during off-peak
4. **Async generation**: Stream response so first token arrives sooner
5. **Reduce k**: Fewer retrieved chunks = less context = faster generation
6. **Smaller models**: Use Haiku/Sonnet for simple queries
7. **Optimize chunk size**: Smaller chunks = fewer tokens in context

Typical p95 latency target: < 2 seconds for complete RAG.

---

### 16.4 Code & Implementation (46-55)

**Q46: Write code to create a Bedrock KB using boto3.**

```python
import boto3
import json

bedrock = boto3.client('bedrock', region_name='us-east-1')
aoss = boto3.client('opensearchserverless')

# 1. Create vector store collection
collection = aoss.create_collection(name='my-kb-collection', type='VECTORSEARCH')
collection_arn = collection['arn']

# 2. Create knowledge base
kb = bedrock.create_knowledge_base(
    name='my-knowledge-base',
    roleArn='arn:aws:iam::123456789:role/kb-role',
    knowledgeBaseConfiguration={
        'type': 'VECTOR',
        'vectorKnowledgeBaseConfiguration': {
            'embeddingModelArn': 'arn:aws:bedrock:us-east-1::embedding-model/amazon.titan-embed-text-v2:0',
        }
    },
    storageConfiguration={
        'type': 'OPENSEARCH_SERVERLESS',
        'opensearchServerlessConfiguration': {
            'collectionArn': collection_arn,
            'vectorIndexName': 'my-index',
        }
    }
)

# 3. Create data source
ds = bedrock.create_data_source(
    knowledgeBaseId=kb['knowledgeBase']['knowledgeBaseId'],
    name='my-s3-source',
    dataSourceConfiguration={
        'type': 'S3',
        's3Configuration': {
            'bucketArn': 'arn:aws:s3:::my-bucket',
        }
    },
    vectorIngestionConfiguration={
        'chunkingConfiguration': {
            'chunkingStrategy': 'HIERARCHICAL',
            'hierarchicalChunkingConfiguration': {
                'levelConfigurations': [
                    {'maxTokens': 500, 'overlapPercentage': 10},
                    {'maxTokens': 2000, 'overlapPercentage': 5},
                ]
            }
        }
    }
)

# 4. Start sync
sync = bedrock.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceConfiguration={'dataSourceId': ds_id})
```

---

**Q47: How do you query Bedrock KB with streaming?**

```python
import boto3
import json

bedrock_runtime = boto3.client('bedrock-agent-runtime')

# First retrieve context
retrieve_resp = bedrock_runtime.retrieve(
    knowledgeBaseId=kb_id,
    retrievalQuery={'text': question},
)
context = '\n\n'.join([r['content']['text'] for r in retrieve_resp['retrievalResults']])

# Then stream generation
prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

response = bedrock_runtime.invoke_model_with_response_stream(
    modelId='anthropic.claude-3-sonnet-4-20250514',
    body=json.dumps({
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 2048,
        'messages': [{'role': 'user', 'content': prompt}]
    })
)

for event in response['body']:
    chunk = json.loads(event['chunk']['bytes'])
    if chunk['type'] == 'content_block_delta':
        print(chunk['delta']['text'], end='', flush=True)
```

---

**Q48: Write a complete custom RAG pipeline using LangChain and Bedrock.**

```python
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import boto3

# Setup
bedrock_runtime = boto3.client('bedrock-runtime')
embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0', region_name='us-east-1')
vectorstore = OpenSearchVectorStore(embedding=embeddings, index_name='my-index')

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# Custom prompt
template = """Use the context to answer. Cite sources.

Context: {context}

Question: {question}

Answer:"""

# Create chain
qa = RetrievalQA.from_chain_type(
    llm=Bedrock(model_id='anthropic.claude-3-sonnet-4-20250514', client=bedrock_runtime),
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': PromptTemplate(template=template, input_variables=['context', 'question'])}
)

# Query
result = qa({'query': 'What is the policy?'})
print(result['result'])
```

---

**Q49: How do you implement query expansion for better retrieval?**

```python
def expand_query(query: str, llm) -> list[str]:
    """Generate alternative query formulations."""
    prompt = f"""Generate 5 different ways to ask this question.
Rephrase using synonyms, different sentence structures, narrower and broader terms.

Query: {query}

Return one variation per line."""
    
    response = llm.invoke(prompt)
    variants = [query] + [v.strip() for v in response.split('\n') if v.strip()]
    return variants[:5]

def expanded_retrieve(query: str, retriever, embedder, llm, k: int = 5):
    """Retrieve using multiple query variants and fuse results."""
    variants = expand_query(query, llm)
    
    all_scores = {}
    for variant in variants:
        emb = embedder.embed_query(variant)
        results = retriever.search(emb, k=k*2)
        
        for rank, r in enumerate(results):
            doc_id = r['doc_id']
            rrf = 1 / (60 + rank + 1)  # RRF with k=60
            all_scores[doc_id] = all_scores.get(doc_id, 0) + rrf
    
    # Sort by fused score
    sorted_ids = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_ids[:k]
```

---

**Q50: Write code to evaluate RAG using RAGAS.**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Prepare test data
data = {
    'question': ['What is the policy?', 'How to reset password?'],
    'answer': ['The policy states...', 'To reset password...'],
    'contexts': [['Policy document text...'], ['Password reset steps...']],
    'ground_truth': ['Policy answer...', 'Password reset answer...'],
}

dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

print(result)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92, ...}
```

---

**Q51: How do you handle PII in documents before embedding?**

```python
import boto3

def detect_and_redact_pii(text: str) -> tuple[str, list]:
    """Detect PII and return redacted text + findings."""
    comprehend = boto3.client('comprehend')
    
    response = comprehend.detect_pii_entities(Text=text, LanguageCode='en')
    entities = response['Entities']
    
    redacted = text
    findings = []
    
    # Sort descending to preserve offsets
    entities = sorted(entities, key=lambda x: x['BeginOffset'], reverse=True)
    
    for entity in entities:
        start = entity['BeginOffset']
        end = entity['EndOffset']
        pii_type = entity['Type']
        
        redacted = redacted[:start] + f'[{pii_type}]' + redacted[end:]
        findings.append({'type': pii_type, 'start': start, 'end': end})
    
    return redacted, findings

# Use in ingestion pipeline
for chunk in chunks:
    chunk.page_content, findings = detect_and_redact_pii(chunk.page_content)
    chunk.metadata['pii_findings'] = findings
```

---

**Q52: Implement async ingestion for large document corpus.**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import boto3

async def ingest_large_corpus(documents: list, max_workers: int = 10):
    """Async parallel ingestion for large document sets."""
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    # Parse all documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    
    # Embed in parallel batches
    embedder = BedrockEmbeddings()
    all_embeddings = []
    
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = [c.page_content for c in all_chunks[i:i+batch_size]]
        
        embeddings = await loop.run_in_executor(
            executor,
            embedder.embed_documents,
            batch
        )
        all_embeddings.extend(embeddings)
    
    # Index to vector store
    vectorstore.add_texts(
        texts=[c.page_content for c in all_chunks],
        embeddings=all_embeddings,
    )
    
    return len(all_chunks)
```

---

**Q53: How do you implement A/B testing for retrievers?**

```python
import hashlib

class ABRetrieverTest:
    def __init__(self, retriever_a, retriever_b, variant_b_ratio=0.5):
        self.retriever_a = retriever_a
        self.retriever_b = retriever_b
        self.variant_b_ratio = variant_b_ratio
    
    def get_variant(self, user_id: str) -> str:
        """Deterministic assignment based on user ID."""
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        return 'b' if hash_val < self.variant_b_ratio * 100 else 'a'
    
    def retrieve(self, query: str, user_id: str):
        variant = self.get_variant(user_id)
        
        if variant == 'a':
            results = self.retriever_a.search(query)
            retriever_name = 'retriever_a'
        else:
            results = self.retriever_b.search(query)
            retriever_name = 'retriever_b'
        
        return {
            'results': results,
            'variant': variant,
            'retriever': retriever_name,
            'user_id': user_id,
        }
    
    def record_feedback(self, user_id: str, feedback: float):
        """Store feedback for later analysis."""
        variant = self.get_variant(user_id)
        # Store in DynamoDB or CloudWatch for analysis
```

---

**Q54: Write a cross-encoder reranking function.**

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-12-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank documents using cross-encoder.
        Jointly encodes [query, document] for more accurate scoring.
        """
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'document': doc, 'score': float(score), 'rank': i + 1}
            for i, (doc, score) in enumerate(scored_docs[:top_k])
        ]

# Usage in pipeline
reranker = CrossEncoderReranker()
candidates = vector_store.search(query_emb, k=50)
docs = [c['content'] for c in candidates]
reranked = reranker.rerank(question, docs, top_k=5)
```

---

**Q55: How do you implement cost tracking for Bedrock RAG?**

```python
import boto3
from datetime import datetime

class RAGCostTracker:
    def __init__(self, cost_explorer_client=None):
        self.ce = cost_explorer_client or boto3.client('costexplorer')
    
    def track_embedding(self, num_tokens: int, model: str = 'titan-v2'):
        """Track embedding API cost."""
        price_per_1k = 0.0002  # Titan v2
        cost = (num_tokens / 1000) * price_per_1k
        self._log('embedding', cost, {'model': model, 'tokens': num_tokens})
        return cost
    
    def track_generation(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ):
        """Track LLM generation cost."""
        pricing = {
            'claude-3-sonnet': {'in': 0.003, 'out': 0.015},
            'claude-3-haiku': {'in': 0.00025, 'out': 0.00125},
        }
        p = pricing.get(model, pricing['claude-3-sonnet'])
        
        cost = (input_tokens / 1000) * p['in'] + (output_tokens / 1000) * p['out']
        self._log('generation', cost, {
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        })
        return cost
    
    def _log(self, cost_type: str, cost: float, metadata: dict):
        """Log cost to CloudWatch."""
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='RAG/Costs',
            MetricData=[{
                'MetricName': f'{cost_type.upper()}Cost',
                'Value': cost,
                'Unit': 'None',
                'Dimensions': [
                    {'Name': 'CostType', 'Value': cost_type},
                    {'Name': 'Model', 'Value': metadata.get('model', 'unknown')},
                ]
            }]
        )
    
    def get_monthly_cost(self, start_date: str, end_date: str) -> dict:
        """Get monthly cost breakdown from Cost Explorer."""
        response = self.ce.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'SERVICE', 'Key': 'SERVICE'}],
        )
        return response
```

---

---

# Part 17: Hands-On Labs

## Lab 1: Build a Local RAG with ChromaDB

**Objective:** Build a complete RAG system using ChromaDB (local vector store) and OpenAI embeddings.

**Duration:** 30-45 minutes

### Lab 1.1 Setup

```bash
# Create virtual environment
python -m venv rag-labs
source rag-labs/bin/activate  # On Windows: rag-labs\Scripts\activate

# Install dependencies
pip install chromadb langchain langchain-openai openai python-dotenv pandas
```

### Lab 1.2 Create the RAG Application

```python
# lab1_rag.py
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LocalRAG:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.llm = OpenAI(temperature=0)
    
    def ingest_documents(self, document_paths: list[str]):
        """Load and index documents."""
        documents = []
        for path in document_paths:
            loader = TextLoader(path)
            documents.extend(loader.load())
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        
        print(f"Indexed {len(chunks)} chunks from {len(document_paths)} documents")
        return len(chunks)
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.vectorstore:
            return "No documents indexed. Run ingest_documents first."
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
        
        return qa_chain.run(question)

# Usage
if __name__ == "__main__":
    rag = LocalRAG(persist_dir="./lab1_chroma")
    
    # Create sample documents
    os.makedirs("./sample_docs", exist_ok=True)
    with open("./sample_docs/policy.txt", "w") as f:
        f.write("""Password Policy
        
        1. Minimum password length is 12 characters.
        2. Passwords must include uppercase, lowercase, numbers, and symbols.
        3. Passwords expire every 90 days.
        4. Cannot reuse the last 10 passwords.
        5. Multi-factor authentication is required for all users.
        """)
    
    # Ingest and query
    rag.ingest_documents(["./sample_docs/policy.txt"])
    answer = rag.query("What is the minimum password length?")
    print(f"Answer: {answer}")
```

### Lab 1.3 Exercises

1. Modify chunk size to 200 and observe retrieval differences
2. Add a second document and verify multi-doc retrieval
3. Implement metadata filtering by adding source metadata

---

## Lab 2: Set Up Bedrock Knowledge Base End-to-End

**Objective:** Create a production-ready Bedrock KB with S3 data source.

**Duration:** 60-90 minutes

### Lab 2.1 Prerequisites

```bash
# AWS CLI configured with appropriate credentials
export AWS_PROFILE=your-profile
export AWS_REGION=us-east-1

# Verify credentials
aws sts get-caller-identity
```

### Lab 2.2 Complete Setup Script

```python
# lab2_bedrock_kb.py
import boto3
import json
import time
import os

class BedrockKBSetup:
    def __init__(self, region: str = "us-east-1"):
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.aoss = boto3.client('opensearchserverless', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
    
    def create_kb_role(self) -> str:
        """Create IAM role for Bedrock KB."""
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        role = self.iam.create_role(
            RoleName="bedrock-kb-lab-role",
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for Bedrock KB Lab"
        )
        role_arn = role['Role']['Arn']
        
        # Attach policies
        self.iam.attach_role_policy(
            RoleName="bedrock-kb-lab-role",
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
        )
        
        self.iam.attach_role_policy(
            RoleName="bedrock-kb-lab-role",
            PolicyArn="arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
        )
        
        # Add OpenSearch policy
        self.iam.put_role_policy(
            RoleName="bedrock-kb-lab-role",
            PolicyName="OpenSearchAccess",
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["aoss:*"], "Resource": "*"}]
            })
        )
        
        # Wait for role to propagate
        time.sleep(10)
        return role_arn
    
    def create_opensearch_collection(self, name: str) -> str:
        """Create OpenSearch Serverless collection."""
        collection = self.aoss.create_collection(
            name=name,
            type="VECTORSEARCH",
            standby_replicas="ENABLED",
        )
        collection_arn = collection['arn']
        collection_id = collection['id']
        
        # Wait for active
        print("Waiting for collection to be active...")
        while True:
            status = self.aoss.get_collection(id=collection_id)['status']
            if status == 'ACTIVE':
                break
            time.sleep(10)
        
        # Create access policy
        self.aoss.create_access_policy(
            name=f"{name}-access",
            policy=json.dumps([{
                "Rules": [
                    {"Resource": [f"collection/{collection_id}"], "Permission": ["aoss:*"]},
                    {"Resource": [f"index/{collection_id}/*"], "Permission": ["aoss:*"]},
                ],
                "Principal": {"AWS": [f"arn:aws:iam::{self.account_id}:role/bedrock-kb-lab-role"]}
            }]),
            type="data"
        )
        
        return collection_arn
    
    def create_knowledge_base(
        self,
        name: str,
        role_arn: str,
        collection_arn: str,
    ) -> str:
        """Create Bedrock Knowledge Base."""
        response = self.bedrock.create_knowledge_base(
            name=name,
            description="Lab 2 Knowledge Base",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": "arn:aws:bedrock:us-east-1::embedding-model/amazon.titan-embed-text-v2:0",
                    "embeddingModelConfiguration": {
                        "bedrockEmbeddingModelConfiguration": {
                            "dimensions": 1536,
                            "embeddingType": "SEMANTIC"
                        }
                    }
                }
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": collection_arn,
                    "vectorIndexName": "bedrock-kb-index",
                    "fieldMapping": {
                        "vectorField": "vector",
                        "textField": "text",
                        "metadataField": "metadata"
                    }
                }
            }
        )
        
        kb_arn = response['knowledgeBase']['knowledgeBaseArn']
        kb_id = response['knowledgeBase']['knowledgeBaseId']
        
        print(f"Knowledge Base created: {kb_id}")
        return kb_id
    
    def create_s3_source(self, kb_id: str, bucket_name: str, prefix: str = "") -> str:
        """Create S3 data source."""
        response = self.bedrock.create_data_source(
            knowledgeBaseId=kb_id,
            name="lab-s3-source",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{bucket_name}",
                    "inclusionPrefixes": [prefix] if prefix else None,
                }
            },
            vectorIngestionConfiguration={
                "chunkingConfiguration": {
                    "chunkingStrategy": "HIERARCHICAL",
                    "hierarchicalChunkingConfiguration": {
                        "levelConfigurations": [
                            {"maxTokens": 500, "overlapPercentage": 10},
                            {"maxTokens": 2000, "overlapPercentage": 5},
                        ]
                    }
                }
            }
        )
        
        return response['dataSource']['dataSourceId']
    
    def run_sync(self, kb_id: str, ds_id: str) -> str:
        """Start ingestion job and wait for completion."""
        response = self.bedrock.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceConfiguration={"dataSourceId": ds_id}
        )
        job_id = response['ingestionJob']['jobId']
        
        print("Sync job started, waiting for completion...")
        while True:
            status = self.bedrock.get_ingestion_job(knowledgeBaseId=kb_id, jobId=job_id)
            job_status = status['ingestionJob']['status']
            
            if job_status == 'COMPLETE':
                print("Sync completed successfully!")
                stats = status['ingestionJob'].get('statistics', {})
                print(f"  Documents: {stats.get('documentsUploaded', 0)}")
                break
            elif job_status == 'FAILED':
                print(f"Sync failed: {status['ingestionJob'].get('failureReason')}")
                break
            
            time.sleep(30)
        
        return job_id

# Usage
if __name__ == "__main__":
    setup = BedrockKBSetup()
    
    # Create KB infrastructure
    role_arn = setup.create_kb_role()
    collection_arn = setup.create_opensearch_collection("lab2-kb-collection")
    kb_id = setup.create_knowledge_base("lab2-kb", role_arn, collection_arn)
    
    # Create data source (use your bucket)
    ds_id = setup.create_s3_source(kb_id, "your-bucket-name", "documents/")
    
    # Run sync
    job_id = setup.run_sync(kb_id, ds_id)
    
    print(f"\nKB ID: {kb_id}")
    print(f"Save this ID for Lab 2.3")
```

### Lab 2.3 Query the Knowledge Base

```python
# lab2_query.py
import boto3
import json

bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

KB_ID = "your-kb-id-here"  # From Lab 2.2 setup

def query_kb(question: str, top_k: int = 5) -> dict:
    """Query Bedrock KB with RAG."""
    response = bedrock_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KB_ID,
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-4-20250514",
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": top_k,
                    }
                },
            },
        },
    )
    
    return {
        "answer": response['output']['text'],
        "citations": [
            {
                "source": c['location']['document'].get('title', 'Unknown'),
                "score": c['location']['score'],
            }
            for c in response.get('citations', [])
        ]
    }

# Test queries
if __name__ == "__main__":
    questions = [
        "What is the password policy?",
        "How do I enable MFA?",
        "What are the security requirements?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        result = query_kb(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['citations']}")
```

---

## Lab 3: Implement Hybrid Search with Custom Re-ranking

**Objective:** Build hybrid retrieval combining dense + sparse search with cross-encoder re-ranking.

```python
# lab3_hybrid_rerank.py
import numpy as np
from rank_bm25 import BM25Okapi
import re

class HybridSearchRerank:
    """
    Hybrid search combining:
    1. Dense (vector) search via embeddings
    2. Sparse (BM25) search
    3. Cross-encoder reranking
    """
    
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = None
        self.corpus = []  # For BM25
    
    def index_documents(self, documents: list[dict]):
        """Index documents for both dense and sparse search."""
        texts = [doc['content'] for doc in documents]
        
        # Dense indexing
        self.vector_store.add_texts(texts, metadatas=[doc.get('metadata') for doc in documents])
        
        # Sparse indexing (BM25)
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = texts
    
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _rrf_fusion(
        self,
        dense_results: list,
        sparse_results: list,
        k: int = 60,
        alpha: float = 0.7,
    ) -> list:
        """Reciprocal Rank Fusion."""
        scores = {}
        
        for rank, r in enumerate(dense_results):
            doc_id = r.get('doc_id', r.get('id', rank))
            scores[doc_id] = scores.get(doc_id, 0) + alpha * (1 / (k + rank + 1))
        
        for rank, r in enumerate(sparse_results):
            doc_id = r.get('doc_id', r.get('id', rank))
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (k + rank + 1))
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def search(
        self,
        query: str,
        k: int = 20,
        alpha: float = 0.7,
    ) -> list[dict]:
        """
        Search with hybrid retrieval and reranking.
        """
        # Dense search
        query_emb = self.embedder.embed_query(query)
        dense_results = self.vector_store.similarity_search(query_emb, k=k*2)
        
        # Sparse search
        tokenized_query = self._tokenize(query)
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_ranked = sorted(enumerate(sparse_scores), key=lambda x: x[1], reverse=True)[:k*2]
        sparse_results = [
            {'doc_id': idx, 'score': score, 'content': self.corpus[idx]}
            for idx, score in sparse_ranked
        ]
        
        # RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results, alpha=alpha)
        
        # Re-rank top candidates with cross-encoder
        top_candidates = []
        for doc_id, _ in fused[:10]:
            # Find the actual document
            for r in dense_results + sparse_results:
                if r.get('doc_id') == doc_id or r.get('id') == doc_id:
                    top_candidates.append(r)
                    break
        
        return top_candidates[:k]

# Cross-encoder reranking
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        return [
            {'content': doc, 'score': float(score), 'rank': i+1}
            for i, (doc, score) in enumerate(results[:top_k])
        ]
```

---

## Lab 4: Build a Multi-Modal RAG Pipeline

**Objective:** Implement RAG that handles documents with images.

```python
# lab4_multimodal_rag.py
import base64
import boto3

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

class MultimodalRAG:
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.bedrock_agent = boto3.client('bedrock-agent-runtime')
    
    def embed_image(self, image_path: str) -> list[float]:
        """Generate embedding for image using Titan Multimodal."""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = bedrock_runtime.invoke_model(
            modelId='amazon.titan-embed-image-v1:0',
            body=json.dumps({
                "inputImage": image_data,
                "inputText": "Describe this image for semantic search"
            }),
            contentType='application/json',
            accept='application/json'
        )
        
        return json.loads(response['body'].read())['embedding']
    
    def ask_about_document_with_images(
        self,
        question: str,
        image_paths: list[str] = None,
    ) -> str:
        """
        Answer questions about documents that include images.
        Uses Claude 3 with vision capability.
        """
        content = []
        
        # Add question
        content.append({"type": "text", "text": question})
        
        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_data
                    }
                })
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{
                "role": "user",
                "content": content
            }]
        })
        
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-sonnet-4-20250514',
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        return json.loads(response['body'].read())['content'][0]['text']
```

---

## Lab 5: Evaluate RAG Quality with RAGAS

**Objective:** Set up automated evaluation pipeline using RAGAS metrics.

```python
# lab5_ragas_eval.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import pandas as pd

class RAGEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    
    def prepare_dataset(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] = None,
    ) -> Dataset:
        """Create RAGAS dataset from evaluation data."""
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        return Dataset.from_dict(data)
    
    def evaluate_dataset(self, dataset: Dataset) -> dict:
        """Run RAGAS evaluation."""
        result = evaluate(dataset, metrics=self.metrics)
        return result
    
    def generate_report(self, result, output_path: str = "eval_report.csv"):
        """Generate evaluation report."""
        df = pd.DataFrame([result])
        df.to_csv(output_path, index=False)
        print(f"Report saved to {output_path}")
        return df

# Usage Example
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # Sample evaluation data
    questions = [
        "What is the password minimum length?",
        "How often must passwords be changed?",
        "Is MFA required?",
    ]
    
    answers = [
        "The minimum password length is 12 characters.",
        "Passwords must be changed every 90 days.",
        "Yes, multi-factor authentication is required.",
    ]
    
    contexts = [
        ["The minimum password length is 12 characters with uppercase, lowercase, numbers, and symbols."],
        ["Passwords expire every 90 days and cannot be reused."],
        ["Multi-factor authentication is required for all users."],
    ]
    
    # Create dataset and evaluate
    dataset = evaluator.prepare_dataset(questions, answers, contexts)
    results = evaluator.evaluate_dataset(dataset)
    
    print("Evaluation Results:")
    print(f"  Faithfulness: {results['faithfulness']:.3f}")
    print(f"  Answer Relevancy: {results['answer_relevancy']:.3f}")
    print(f"  Context Precision: {results['context_precision']:.3f}")
    print(f"  Context Recall: {results['context_recall']:.3f}")
```

---

# Part 18: Appendices

## Appendix A: Glossary of Terms

| Term | Definition |
|------|------------|
| **ANN** | Approximate Nearest Neighbor — algorithms for fast similarity search |
| **BM25** | Best Matching 25 — sparse retrieval algorithm based on term frequency |
| **Chunk** | Small piece of a document optimized for retrieval |
| **Cross-encoder** | Neural network that jointly encodes query-document pairs |
| **Dense retrieval** | Vector-based semantic search using neural embeddings |
| **Embedding** | Dense vector representation of text in high-dimensional space |
| **HNSW** | Hierarchical Navigable Small World — ANN algorithm for vector search |
| **Hybrid search** | Combining dense (semantic) and sparse (keyword) retrieval |
| **HyDE** | Hypothetical Document Embeddings — query expansion technique |
| **KB** | Knowledge Base (in Bedrock context) |
| **LLM** | Large Language Model |
| **MMR** | Maximum Marginal Relevance — diversity in retrieval results |
| **RAG** | Retrieval-Augmented Generation — architecture combining retrieval + generation |
| **RAGAS** | RAG Assessment — LLM-based evaluation framework |
| **Re-ranking** | Secondary scoring of retrieved candidates for better ordering |
| **RRF** | Reciprocal Rank Fusion — combining rankings from multiple retrievers |
| **Sparse retrieval** | Traditional IR using term frequency (BM25, TF-IDF) |
| **Vector store** | Database optimized for storing and searching vector embeddings |

## Appendix B: API Reference Quick-Reference

### Bedrock Knowledge Bases APIs

```python
# Create Knowledge Base
bedrock.create_knowledge_base(
    name=str,
    roleArn=str,
    knowledgeBaseConfiguration=dict,  # type, embedding model
    storageConfiguration=dict,  # vector store config
)

# Create Data Source
bedrock.create_data_source(
    knowledgeBaseId=str,
    name=str,
    dataSourceConfiguration=dict,  # type, S3/Confluence config
    vectorIngestionConfiguration=dict,  # chunking config
)

# Start Sync
bedrock.start_ingestion_job(
    knowledgeBaseId=str,
    dataSourceConfiguration=dict,
)

# Retrieve (custom RAG)
bedrock_agent_runtime.retrieve(
    knowledgeBaseId=str,
    retrievalQuery=dict,  # {text: query}
    retrievalConfiguration=dict,  # numberOfResults, filter, etc.
)

# RetrieveAndGenerate (managed RAG)
bedrock_agent_runtime.retrieve_and_generate(
    input=dict,  # {text: question}
    retrieveAndGenerateConfiguration=dict,  # KB config, model, etc.
)
```

### Common Response Structures

```python
# RetrieveAndGenerate response
{
    "output": {"text": str},
    "citations": [
        {
            "generatedResponsePart": {"textSegment": str},
            "location": {
                "type": "CHUNK",
                "score": float,
                "document": {"uri": str, "title": str}
            }
        }
    ],
    "retrievalDetails": {"retrievalIds": list, "knowledgeBaseId": str}
}

# Ingestion job status
{
    "ingestionJob": {
        "jobId": str,
        "status": "COMPLETE|FAILED|IN_PROGRESS",
        "statistics": {
            "documentsScanned": int,
            "documentsUploaded": int,
            "documentsFailed": int,
        },
        "failureReason": str  # if FAILED
    }
}
```

## Appendix C: Terraform Snippets for Bedrock KB

```hcl
# Terraform configuration for Bedrock Knowledge Base
# Requires AWS provider >= 5.0

variable "region" {
  default = "us-east-1"
}

# IAM Role for KB
resource "aws_iam_role" "bedrock_kb_role" {
  name = "bedrock-kb-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "bedrock.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "bedrock_kb_s3" {
  role       = aws_iam_role.bedrock_kb_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_role_policy" "bedrock_kb_aoss" {
  name = "bedrock-kb-aoss"
  role = aws_iam_role.bedrock_kb_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["aoss:*"]
      Resource = "*"
    }]
  })
}

# OpenSearch Serverless Collection
resource "aws_opensearchserverless_collection" "kb_collection" {
  name       = "bedrock-kb-collection"
  type       = "VECTORSEARCH"
  standby_replicas = "ENABLED"
}

# Bedrock Knowledge Base
resource "aws_bedrock_knowledge_base" "main" {
  name        = "main-knowledge-base"
  role_arn    = aws_iam_role.bedrock_kb_role.arn

  knowledge_base_configuration {
    type = "VECTOR"

    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:us-east-1::embedding-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"

    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.kb_collection.arn
      vector_index_name = "bedrock-kb-index"

      field_mapping {
        vector_field   = "vector"
        text_field     = "text"
        metadata_field = "metadata"
      }
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.bedrock_kb_s3,
    aws_iam_role_policy.bedrock_kb_aoss
  ]
}

# S3 Data Source
resource "aws_bedrock_data_source" "main" {
  name               = "main-data-source"
  knowledge_base_id  = aws_bedrock_knowledge_base.main.id

  data_source_configuration {
    type = "S3"

    s3_configuration {
      bucket_arn = "arn:aws:s3:::your-bucket-name"
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "HIERARCHICAL"

      hierarchical_chunking_configuration {
        level_configurations {
          max_tokens         = 500
          overlap_percentage = 10
        }
        level_configurations {
          max_tokens         = 2000
          overlap_percentage = 5
        }
      }
    }
  }
}
```

## Appendix D: Cost Calculator Reference

### Component Cost Breakdown (us-east-1)

| Component | Unit | Cost |
|-----------|------|------|
| **Titan Embeddings v2** | per 1K tokens | $0.0002 |
| **Titan Embeddings v1** | per 1K tokens | $0.0001 |
| **Claude 3.5 Sonnet (input)** | per 1K tokens | $0.003 |
| **Claude 3.5 Sonnet (output)** | per 1K tokens | $0.015 |
| **Claude 3 Haiku (input)** | per 1K tokens | $0.00025 |
| **Claude 3 Haiku (output)** | per 1K tokens | $0.00125 |
| **OpenSearch Serverless (storage)** | per GB/month | $0.24 |
| **OpenSearch Serverless (compute)** | per OCU/hour | $0.024 |
| **S3 Standard (storage)** | per GB/month | $0.023 |

### Example Cost Calculation

**Scenario:** 10,000 documents, 5,000 chars each (1,250 tokens), 10 chunks per doc, 1,000 daily queries

```
INGESTION COSTS:
- Total chunks: 100,000
- Embedding tokens: 100,000 × 1,250 = 125M tokens
- Embedding cost: 125M ÷ 1,000 × $0.0002 = $25
- OpenSearch storage (6KB/vector × 100K): 600MB = ~$0.15/month

QUERY COSTS (daily):
- Query embedding: 1,000 × 50 tokens = 50K tokens = $0.01/day = $0.30/month
- LLM input: 1,000 × (50 query + 5,000 context) = 5.05M tokens = $15.15/month
- LLM output: 1,000 × 500 tokens = 500K tokens = $7.50/month

TOTAL MONTHLY: ~$48 + compute (~$20-40 for moderate usage) = ~$70-90/month
```

## Appendix E: Further Reading and Resources

### AWS Official Documentation

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
- [Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan.html)
- [OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)

### Research Papers

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Original RAG paper
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)
- [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)

### Blog Posts and Tutorials

- [Bedrock KB Deep Dive Blog](https://aws.amazon.com/blogs/machine-learning/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database-comparison/)

### Open Source Tools

- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [Chroma](https://github.com/chroma-core/chroma) — Vector database for embeddings
- [LangChain](https://github.com/langchain-ai/langchain) — LLM application framework
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation framework
- [LLM-AutoEval](https://github.com/aweirdwizard/LLM-AutoEval) — Automated LLM evaluation

---

## Document Summary

This guide has covered the complete landscape of Retrieval-Augmented Generation (RAG) with Amazon Bedrock Knowledge Bases:

| Part | Topic | Key Takeaways |
|------|-------|---------------|
| 0-1 | Foundations | RAG solves LLM hallucination, stale knowledge, and private data access |
| 2 | Embeddings | Vectors enable semantic search; cosine similarity finds related content |
| 3 | Chunking | Chunk size determines retrieval granularity; semantic > fixed-size |
| 4 | Vector DBs | HNSW enables fast ANN search; OpenSearch Serverless is Bedrock's default |
| 5 | RAG Pipeline | Ingestion (load→chunk→embed→index) and Query (embed→retrieve→generate) |
| 6 | Retrieval | Hybrid search + re-ranking improves quality; MMR adds diversity |
| 7 | Context | Prompt engineering and chunk ordering affect generation quality |
| 8 | Bedrock KB | Fully managed RAG with S3/Confluence/SharePoint/Salesforce sources |
| 9 | Titan Embeddings | AWS-native embeddings with Matryoshka dimension reduction |
| 10 | Code Examples | Complete working code for boto3, LangChain, and streaming |
| 11 | Advanced Patterns | Self-RAG, CRAG, Graph RAG, Agentic RAG for production |
| 12 | Evaluation | RAGAS metrics, retrieval/generation metrics, human evaluation |
| 13 | Production Ops | Monitoring, latency optimization, cost tracking, A/B testing |
| 14 | Security | PII handling, encryption, IAM, audit logging, compliance |
| 15 | Troubleshooting | Debug retrieval, generation, performance, infrastructure |
| 16 | Interview Q&A | 50+ questions from fundamentals to architecture design |
| 17 | Hands-On Labs | 5 practical labs from local ChromaDB to Bedrock KB to RAGAS eval |
| 18 | Appendices | Glossary, API reference, Terraform, cost calculator |

**Total document length: ~18,000+ words**

---

*Document Version 1.0 — June 2025*
*This guide is intended for educational and reference purposes. AWS service availability and pricing subject to change; verify current details in official AWS documentation.*