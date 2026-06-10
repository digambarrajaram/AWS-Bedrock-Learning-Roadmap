# AWS Bedrock & AI Orchestration — Complete Python Learning Guide

> **Goal:** Master Python concepts, libraries, and design patterns to connect to and orchestrate AI models using AWS Bedrock, LangChain, and LangGraph.

---

## Table of Contents

1. [Python Core Concepts](#1-python-core-concepts)
   - 1.1 Classes & Inheritance
   - 1.2 Asyncio (async/await)
   - 1.3 Type Hinting
   - 1.4 Decorators
2. [Data Validation & Config](#2-data-validation--config)
   - 2.1 Pydantic V2 BaseModel
   - 2.2 Pydantic Field & Validation
   - 2.3 python-dotenv & OS Environment
3. [AWS Bedrock & Integration](#3-aws-bedrock--integration)
   - 3.1 IAM & Authentication
   - 3.2 boto3 (AWS SDK for Python)
   - 3.3 botocore.exceptions
   - 3.4 Bedrock Converse API
   - 3.5 Streaming Responses
4. [LangChain Ecosystem](#4-langchain-ecosystem)
   - 4.1 LCEL & Runnable Interface
   - 4.2 ChatBedrock (langchain-aws)
   - 4.3 Document Loaders (langchain-community)
   - 4.4 Prompt Templates
5. [LangGraph (Multi-Agent)](#5-langgraph-multi-agent)
   - 5.1 StateGraph Architecture
   - 5.2 Nodes (Python Functions)
   - 5.3 Edges & Conditional Routing
   - 5.4 Graph Memory (Checkpointers)
6. [Vector Databases & RAG](#6-vector-databases--rag)
   - 6.1 Text Splitters (Chunking)
   - 6.2 Bedrock Embeddings
   - 6.3 Vector Store Clients (Chroma / Pinecone)
   - 6.4 Full RAG Pipeline
7. [Prompt Engineering in Code](#7-prompt-engineering-in-code)
8. [Error Handling Patterns](#8-error-handling-patterns)
9. [Production Patterns & Best Practices](#9-production-patterns--best-practices)
10. [Learning Roadmap](#10-learning-roadmap)

---

## 1. Python Core Concepts

### 1.1 Classes & Inheritance

**Why required:** Every major library (LangChain, Pydantic, LangGraph) uses class-based design. You extend base classes to define custom agents, tools, loaders, and state schemas.

**How much required:** Solid understanding of `__init__`, method overriding, `super()`, and abstract base classes.

**Syntax:**
```python
from abc import ABC, abstractmethod

class BaseLLMAgent(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_id})"


class BedrockAgent(BaseLLMAgent):
    def __init__(self, model_id: str, region: str = "us-east-1"):
        super().__init__(model_id)       # Call parent __init__
        self.region = region

    def invoke(self, prompt: str) -> str:
        # concrete implementation
        return f"[{self.model_id}] response to: {prompt}"
```

**Real-world use:** LangChain's `BaseTool` is an abstract class you subclass to create custom tools for your agent:
```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="search query string")

class ProductSearchTool(BaseTool):
    name: str = "product_search"
    description: str = "Search product catalog by name or SKU"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        # call your DB or API
        return f"Results for: {query}"
```

---

### 1.2 Asyncio (async/await)

**Why required:** Bedrock API calls are I/O-bound. With `asyncio`, you can call multiple models or tools concurrently without blocking — critical for multi-agent systems and streaming responses.

**How much required:** Understand `async def`, `await`, `asyncio.gather()` for concurrent calls, and `asyncio.run()` as the entry point.

**Syntax:**
```python
import asyncio

async def call_model(prompt: str, delay: float) -> str:
    await asyncio.sleep(delay)          # simulate I/O (non-blocking)
    return f"Response to: {prompt}"

async def main():
    # Run two model calls concurrently
    results = await asyncio.gather(
        call_model("What is RAG?", 1.0),
        call_model("Explain embeddings", 0.5),
    )
    for r in results:
        print(r)

asyncio.run(main())
```

**Real-world use:** Invoke multiple Bedrock models in parallel for ensemble responses or A/B testing:
```python
import asyncio
import boto3

async def invoke_bedrock_async(client, model_id: str, prompt: str) -> dict:
    loop = asyncio.get_event_loop()
    # boto3 is sync — run in executor so it doesn't block the event loop
    response = await loop.run_in_executor(
        None,
        lambda: client.invoke_model(
            modelId=model_id,
            body=json.dumps({"prompt": prompt, "max_tokens": 500})
        )
    )
    return {"model": model_id, "response": response}

async def compare_models(prompt: str):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    models = ["amazon.titan-text-express-v1", "anthropic.claude-3-haiku-20240307-v1:0"]
    tasks = [invoke_bedrock_async(client, m, prompt) for m in models]
    return await asyncio.gather(*tasks)
```

---

### 1.3 Type Hinting

**Why required:** Pydantic, LangChain, and LangGraph rely heavily on type hints for validation, serialization, and graph state definitions. Without them, state schemas fail.

**How much required:** Know built-in types, `Optional`, `Union`, `List`, `Dict`, `Callable`, `Literal`, and `TypedDict`.

**Syntax:**
```python
from typing import Optional, List, Dict, Literal, TypedDict, Callable, Union

def process_messages(
    messages: List[Dict[str, str]],
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    temperature: float = 0.7,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    ...

# TypedDict — used extensively in LangGraph state
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    context: Optional[str]
    next_step: Literal["search", "answer", "end"]
    iteration: int
```

**Real-world use:** LangGraph state must be `TypedDict` or a Pydantic model:
```python
from typing import Annotated
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]   # Annotated tells LangGraph how to merge
    retrieved_docs: list[str]
    final_answer: Optional[str]
```

---

### 1.4 Decorators

**Why required:** LangChain uses `@tool` to convert a Python function into an agent tool. Testing uses `@patch` to mock AWS calls. You'll also write retry decorators for Bedrock throttling.

**How much required:** Know how to use built-in decorators and write simple custom ones using `functools.wraps`.

**Syntax:**
```python
import functools
import time

# Custom retry decorator for Bedrock throttling
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)          # preserves function name/docstring
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))  # exponential backoff
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def call_bedrock(prompt: str) -> str:
    ...
```

**`@tool` in LangChain:**
```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Use when user asks about weather."""
    return f"Weather in {city}: 28°C, sunny"

# LangChain reads the docstring as the tool description for the LLM
print(get_weather.name)         # "get_weather"
print(get_weather.description)  # docstring above
```

**`@patch` in unit tests:**
```python
from unittest.mock import patch, MagicMock

@patch("boto3.client")
def test_bedrock_call(mock_client):
    mock_client.return_value.invoke_model.return_value = {
        "body": MagicMock(read=lambda: b'{"completion": "Hello"}')
    }
    result = call_bedrock("Say hello")
    assert "Hello" in result
```

---

## 2. Data Validation & Config

### 2.1 Pydantic V2 BaseModel

**Why required:** Pydantic is the backbone of LangChain and LangGraph. Every tool input, chain output, and config object is a Pydantic model. It validates types at runtime and auto-generates JSON schemas (used to describe tools to LLMs).

**How much required:** Know `BaseModel`, field definition, `model_validate`, `model_dump`, and `model_json_schema`.

**Syntax:**
```python
from pydantic import BaseModel, Field
from typing import Optional, List

class LLMConfig(BaseModel):
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 1024
    region: str = "us-east-1"
    stop_sequences: Optional[List[str]] = None

# Instantiate — Pydantic validates types automatically
config = LLMConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.5)

# Serialize to dict (for passing to APIs)
print(config.model_dump())
# {'model_id': '...', 'temperature': 0.5, 'max_tokens': 1024, ...}

# Serialize to JSON string
print(config.model_dump_json())

# Deserialize from dict
config2 = LLMConfig.model_validate({"model_id": "...", "temperature": "0.3"})
# "0.3" (string) is automatically coerced to float 0.3
```

---

### 2.2 Pydantic Field & Validation

**Why required:** LangChain generates JSON schemas from Pydantic models to describe tool arguments to the LLM. The `description` in `Field(...)` becomes the docstring the LLM reads to understand what value to pass.

**Syntax:**
```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal

class BedrockRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temp")
    model_id: str = Field(default="anthropic.claude-3-haiku-20240307-v1:0")
    mode: Literal["chat", "embed", "stream"] = "chat"

    @field_validator("prompt")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def check_embed_no_temperature(self) -> "BedrockRequest":
        if self.mode == "embed" and self.temperature != 0.7:
            raise ValueError("Embedding mode does not use temperature")
        return self

# Pydantic raises ValidationError on bad input
try:
    r = BedrockRequest(prompt="", temperature=2.5)
except Exception as e:
    print(e)   # prompt too short, temperature too high
```

---

### 2.3 python-dotenv & OS Environment

**Why required:** AWS credentials, model IDs, and API keys must never be hardcoded. `python-dotenv` loads them from a `.env` file into `os.environ` at runtime.

**How much required:** Know `load_dotenv()`, `os.getenv()`, and how to set up a `.env` file safely (add to `.gitignore`).

**Setup:**
```bash
pip install python-dotenv
```

**.env file** (never commit this):
```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
PINECONE_API_KEY=...
```

**Usage:**
```python
import os
from dotenv import load_dotenv

load_dotenv()   # loads .env into os.environ

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")    # with fallback
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

if not MODEL_ID:
    raise EnvironmentError("BEDROCK_MODEL_ID not set in environment")
```

**Real-world use — Config class pattern (recommended):**
```python
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class AppConfig(BaseModel):
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model_id: str = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    embedding_model: str = "amazon.titan-embed-text-v1"
    chroma_persist_dir: str = "./chroma_db"

config = AppConfig()
# Now pass `config` everywhere — single source of truth
```

---

## 3. AWS Bedrock & Integration

### 3.1 IAM & Authentication

**Why required:** Before writing a single line of Bedrock code, you need the right IAM permissions. `boto3` uses credentials from the AWS credentials chain.

**How much required:** Understand AWS credentials chain, IRSA for EKS (your background), and model access enablement in Bedrock console.

**Credentials resolution order (boto3):**
1. Explicit code credentials (avoid)
2. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. `~/.aws/credentials` file
4. IAM Role (EC2 / Lambda / EKS via IRSA)
5. AWS SSO

**Required IAM policy for Bedrock:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/*"
    }
  ]
}
```

**Important:** You must also manually enable each model in the AWS Bedrock console under **Model Access** before your code can use it.

---

### 3.2 boto3 (AWS SDK for Python)

**Why required:** `boto3` is the official Python SDK to interact with all AWS services. For Bedrock, you need two clients: `bedrock` (list models, manage) and `bedrock-runtime` (invoke models).

**Install:**
```bash
pip install boto3
```

**Syntax:**
```python
import boto3
import json

# Create client — uses credentials chain automatically
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Direct InvokeModel call (low-level)
response = client.invoke_model(
    modelId="amazon.titan-text-express-v1",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "inputText": "What is machine learning?",
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.7,
        }
    })
)

result = json.loads(response["body"].read())
print(result["results"][0]["outputText"])
```

**Real-world tip — use Sessions for multi-region or role assumption:**
```python
session = boto3.Session(
    region_name="us-west-2",
    profile_name="bedrock-prod"   # from ~/.aws/config
)
client = session.client("bedrock-runtime")
```

---

### 3.3 botocore.exceptions

**Why required:** Bedrock throttles requests, models can be unavailable, and credentials can expire. You must catch specific exceptions instead of bare `except Exception` for clean error handling.

**Syntax:**
```python
from botocore.exceptions import (
    ClientError,
    EndpointResolutionError,
    NoCredentialsError,
    BotoCoreError
)
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

try:
    response = client.invoke_model(modelId="...", body=b"{}")
except NoCredentialsError:
    print("AWS credentials not found — check ~/.aws/credentials or IAM role")
except ClientError as e:
    error_code = e.response["Error"]["Code"]
    if error_code == "ThrottlingException":
        print("Rate limited — retry with exponential backoff")
    elif error_code == "ModelNotReadyException":
        print("Model not enabled — go to Bedrock console > Model Access")
    elif error_code == "ValidationException":
        print(f"Bad request: {e.response['Error']['Message']}")
    elif error_code == "AccessDeniedException":
        print("IAM policy missing bedrock:InvokeModel permission")
    else:
        raise
except BotoCoreError as e:
    print(f"Low-level AWS error: {e}")
```

---

### 3.4 Bedrock Converse API

**Why required:** The **Converse API** is the modern, model-agnostic way to call Bedrock. Unlike `invoke_model` (which requires model-specific JSON bodies), `converse` uses a unified message format that works across Claude, Titan, Llama, Mistral, etc.

**Syntax:**
```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.converse(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    system=[{"text": "You are a helpful cloud infrastructure assistant."}],
    messages=[
        {"role": "user", "content": [{"text": "Explain AWS VPC in one paragraph."}]}
    ],
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.7,
        "topP": 0.9,
    }
)

output = response["output"]["message"]["content"][0]["text"]
print(output)
print(f"Input tokens: {response['usage']['inputTokens']}")
print(f"Output tokens: {response['usage']['outputTokens']}")
```

**Multi-turn conversation (chat history):**
```python
messages = []

def chat(user_input: str) -> str:
    messages.append({
        "role": "user",
        "content": [{"text": user_input}]
    })
    response = client.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        inferenceConfig={"maxTokens": 1024}
    )
    assistant_msg = response["output"]["message"]
    messages.append(assistant_msg)       # add to history for next turn
    return assistant_msg["content"][0]["text"]

print(chat("What is a subnet?"))
print(chat("How does it relate to VPCs?"))   # model remembers context
```

---

### 3.5 Streaming Responses

**Why required:** Large language model responses can take seconds. Streaming lets you display tokens as they arrive, improving perceived latency significantly in chat UIs.

**How much required:** Know `converse_stream` and how to iterate the event stream.

**Syntax:**
```python
def stream_response(prompt: str) -> None:
    response = client.converse_stream(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1024}
    )
    stream = response.get("stream")
    if stream:
        for event in stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    print(delta["text"], end="", flush=True)
            elif "messageStop" in event:
                print()   # newline at end
                stop_reason = event["messageStop"]["stopReason"]
                print(f"\n[Stop reason: {stop_reason}]")

stream_response("Write a Terraform module for an S3 bucket with versioning enabled.")
```

---

## 4. LangChain Ecosystem

### 4.1 LCEL & Runnable Interface

**Why required:** LCEL (LangChain Expression Language) is the composable pipeline system. Chains are built by piping runnables together using `|`. Understanding this is fundamental to building any LangChain application.

**Install:**
```bash
pip install langchain-core langchain-aws langchain-community
```

**Core concept — every component is a Runnable:**
```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# Simple function as runnable
double = RunnableLambda(lambda x: x * 2)
add_ten = RunnableLambda(lambda x: x + 10)

# Chain with pipe operator
chain = double | add_ten
print(chain.invoke(5))   # (5*2)+10 = 20

# Passthrough — pass input unchanged alongside other data
chain = RunnableParallel({
    "original": RunnablePassthrough(),
    "doubled": double
})
print(chain.invoke(5))   # {"original": 5, "doubled": 10}
```

**Real-world RAG chain:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""
Answer based on the context below.
Context: {context}
Question: {question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke("What is the return policy?")
```

---

### 4.2 ChatBedrock (langchain-aws)

**Why required:** `ChatBedrock` wraps the boto3 Bedrock calls behind LangChain's unified chat interface. You get streaming, callbacks, token tracking, and LCEL compatibility for free.

**Syntax:**
```python
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 1024,
    }
)

# Simple invoke
response = llm.invoke("Explain Kubernetes namespaces in simple terms")
print(response.content)

# With system message
messages = [
    SystemMessage(content="You are a DevOps expert. Be concise."),
    HumanMessage(content="What is the difference between a Deployment and a StatefulSet?")
]
response = llm.invoke(messages)
print(response.content)

# Streaming with LangChain
for chunk in llm.stream("Generate a Helm values.yaml for nginx"):
    print(chunk.content, end="", flush=True)
```

---

### 4.3 Document Loaders (langchain-community)

**Why required:** Before building RAG, you need to load documents from PDFs, S3, web pages, or databases into LangChain `Document` objects.

**Syntax:**
```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    S3FileLoader,
    DirectoryLoader
)

# Load PDF
loader = PyPDFLoader("aws_well_architected.pdf")
docs = loader.load()       # list of Document objects
print(docs[0].page_content[:200])
print(docs[0].metadata)    # {"source": "aws_well_architected.pdf", "page": 0}

# Load all .txt files from a directory
loader = DirectoryLoader("./docs/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# Load from S3
loader = S3FileLoader(bucket="my-rag-bucket", key="docs/policy.pdf")
docs = loader.load()

# Load from web URL
loader = WebBaseLoader("https://docs.aws.amazon.com/bedrock/latest/userguide/")
docs = loader.load()
```

---

### 4.4 Prompt Templates

**Why required:** Prompts are the primary interface to LLMs. Templates let you inject dynamic variables cleanly and reuse prompt structure across your application.

**Syntax:**
```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder
)

# Chat prompt with system + human
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])

# Fill template
messages = chat_prompt.format_messages(
    role="senior AWS architect",
    language="English",
    question="What is AWS Bedrock?"
)

# With conversation history (MessagesPlaceholder for chat memory)
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),   # injected at runtime
    ("human", "{input}"),
])
```

---

## 5. LangGraph (Multi-Agent)

### 5.1 StateGraph Architecture

**Why required:** LangGraph lets you build multi-step, stateful AI workflows with branching logic, loops, and parallelism. It's the standard framework for building agents that need to reason across multiple steps (search → analyze → answer → verify).

**Core concepts:**

| Concept | What it is |
|---|---|
| `State` | A TypedDict holding all data flowing through the graph |
| `Node` | A Python function that reads and modifies state |
| `Edge` | Connection between nodes (fixed or conditional) |
| `Graph` | The wired-up network of nodes and edges |
| `Checkpointer` | Persists state between invocations (memory) |

**Install:**
```bash
pip install langgraph
```

**Minimal example:**
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    step: int

graph_builder = StateGraph(State)
# ... add nodes and edges ...
graph = graph_builder.compile()
result = graph.invoke({"messages": [], "step": 0})
```

---

### 5.2 Nodes (Python Functions)

**Why required:** Each node is a unit of work in your agent. It receives the current state, does something (call LLM, search database, run tool), and returns a dict of state updates.

**Syntax:**
```python
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
llm_with_tools = llm.bind_tools(tools)   # attach tools to the LLM

def call_llm_node(state: State) -> dict:
    """Node: calls the LLM with current message history."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}    # LangGraph merges this into state

def tool_executor_node(state: State) -> dict:
    """Node: executes any tool calls the LLM requested."""
    from langgraph.prebuilt import ToolNode
    # ToolNode is a prebuilt node that auto-executes tool calls
    pass

def retrieve_context_node(state: State) -> dict:
    """Node: retrieves relevant documents for RAG."""
    last_message = state["messages"][-1].content
    docs = retriever.invoke(last_message)
    context = "\n\n".join([d.page_content for d in docs])
    return {"context": context}

# Register nodes
graph_builder.add_node("llm", call_llm_node)
graph_builder.add_node("retrieve", retrieve_context_node)
```

---

### 5.3 Edges & Conditional Routing

**Why required:** Edges define the flow. Conditional edges let the agent decide its own next step based on the current state — this is what makes an "agent" different from a simple chain.

**Syntax:**
```python
from langgraph.graph import END
from langchain_core.messages import AIMessage

def route_after_llm(state: State) -> str:
    """Conditional edge: route based on whether LLM called a tool."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"           # go to tool executor
    return "end"                 # done, return to user

# Fixed edge (always goes A → B)
graph_builder.add_edge("retrieve", "llm")

# Conditional edge (route function decides)
graph_builder.add_conditional_edges(
    "llm",                          # from this node
    route_after_llm,                # call this function
    {
        "tools": "tool_executor",   # if returns "tools", go here
        "end": END                  # if returns "end", terminate
    }
)

graph_builder.set_entry_point("retrieve")
graph = graph_builder.compile()
```

**Real-world ReAct agent routing:**
```
User → [retrieve] → [llm] → needs tool? → YES → [tool_executor] → [llm] → loop
                           → NO → END
```

---

### 5.4 Graph Memory (Checkpointers)

**Why required:** By default, graphs are stateless — each `invoke()` starts fresh. Checkpointers persist state across calls using a `thread_id`, enabling multi-turn conversations and resumable workflows.

**Syntax:**
```python
from langgraph.checkpoint.memory import MemorySaver      # in-memory (dev/test)
from langgraph.checkpoint.sqlite import SqliteSaver       # SQLite (single server)
# from langgraph.checkpoint.postgres import PostgresSaver  # Postgres (production)

# In-memory checkpointer
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# thread_id groups messages into a "conversation"
config = {"configurable": {"thread_id": "user-123-session-456"}}

# First turn
result1 = graph.invoke(
    {"messages": [HumanMessage("What is VPC peering?")]},
    config=config
)

# Second turn — graph remembers first turn automatically
result2 = graph.invoke(
    {"messages": [HumanMessage("How does it differ from Transit Gateway?")]},
    config=config   # same thread_id
)

# Inspect full state at any point
state = graph.get_state(config)
print(state.values["messages"])   # full message history
```

---

## 6. Vector Databases & RAG

### 6.1 Text Splitters (Chunking)

**Why required:** LLMs have context limits. Documents must be split into smaller chunks before embedding. Chunk size and overlap directly impact retrieval quality.

**Why not just split by character count?** Semantic meaning crosses sentence boundaries. `RecursiveCharacterTextSplitter` tries to split on paragraph → sentence → word → character boundaries, preserving meaning.

**Syntax:**
```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # target chunk size in characters
    chunk_overlap=200,      # overlap between chunks (preserves context at boundaries)
    separators=["\n\n", "\n", ". ", " ", ""],   # try these in order
    length_function=len
)

docs = loader.load()
chunks = splitter.split_documents(docs)

print(f"Original docs: {len(docs)}")
print(f"Chunks: {len(chunks)}")
print(f"First chunk:\n{chunks[0].page_content[:300]}")
print(f"Metadata: {chunks[0].metadata}")
```

**Chunking guidelines:**

| Use case | chunk_size | chunk_overlap |
|---|---|---|
| Q&A over docs | 500–1000 | 100–200 |
| Summarization | 2000–4000 | 200–500 |
| Code search | 1500–2000 | 300 |
| Short FAQs | 200–500 | 50 |

---

### 6.2 Bedrock Embeddings

**Why required:** Embeddings are numerical representations of text. Similar texts have embeddings that are close together in vector space. You use embeddings to find chunks relevant to a user query.

**Syntax:**
```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1"
)

# Embed a single query
vector = embeddings.embed_query("What is the refund policy?")
print(f"Embedding dimensions: {len(vector)}")    # 1536 for Titan

# Embed multiple documents (batched)
texts = ["Document 1 content", "Document 2 content"]
vectors = embeddings.embed_documents(texts)
print(f"Shape: {len(vectors)} x {len(vectors[0])}")
```

**Alternative: Amazon Titan Embed V2 (better for multilingual):**
```python
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",  # newer, 1024 dims
    region_name="us-east-1"
)
```

---

### 6.3 Vector Store Clients (Chroma / Pinecone)

**Why required:** Vector stores index embeddings and enable fast similarity search. Chroma is great for local dev; Pinecone is managed and scales to billions of vectors.

**Chroma (local development):**
```bash
pip install chromadb langchain-chroma
```

```python
from langchain_chroma import Chroma

# Create store from documents (embeds and indexes automatically)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",    # saves to disk
    collection_name="product_docs"
)

# Load existing store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="product_docs"
)

# Similarity search
results = vectorstore.similarity_search("return policy", k=4)
for doc in results:
    print(doc.page_content[:200])
    print(doc.metadata)

# As retriever (for use in chains)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

**Pinecone (production):**
```bash
pip install pinecone-client langchain-pinecone
```

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if not exists
if "rag-index" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-index",
        dimension=1536,                    # must match embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index("rag-index")

vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
vectorstore.add_documents(chunks)

# Search
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---

### 6.4 Full RAG Pipeline

**Why required:** Combining all the above — loader, splitter, embeddings, vector store, LLM, and prompt — into a complete RAG system.

**Architecture:**
```
[Documents] → [Loader] → [Splitter] → [Embeddings] → [VectorStore]
                                                             ↓
[User Query] → [Embed Query] → [Similarity Search] → [Top-K Chunks]
                                                             ↓
                              [Prompt Template: context + query]
                                                             ↓
                                               [LLM (ChatBedrock)]
                                                             ↓
                                                   [Final Answer]
```

**Complete implementation:**
```python
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import boto3

# 1. Load & split documents
loader = PyPDFLoader("knowledge_base.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 2. Embed & store
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. LLM
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0.3}
)

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")

# 5. Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query
answer = rag_chain.invoke("What is the warranty period for product X?")
print(answer)
```

---

## 7. Prompt Engineering in Code

**Why required:** How you write prompts directly determines response quality. These patterns are used inside your chain/graph code, not just in chat UIs.

**System prompts — define persona and constraints:**
```python
SYSTEM_PROMPT = """
You are a senior AWS Solutions Architect. You help developers design secure,
cost-effective cloud infrastructure.

Rules:
- Always recommend the Well-Architected Framework pillars
- Cite specific AWS services with their exact names
- Flag security risks explicitly
- If unsure, say so — do not hallucinate service features
"""
```

**Few-shot prompting — teach the format via examples:**
```python
FEW_SHOT_PROMPT = """
Extract infrastructure details from the text as JSON.

Examples:
Text: "Deploy a 3-tier app with ALB, ECS on Fargate, and Aurora PostgreSQL"
Output: {{"load_balancer": "ALB", "compute": "ECS Fargate", "database": "Aurora PostgreSQL"}}

Text: "Run ML workloads on EC2 p3.2xlarge with S3 for data storage"
Output: {{"compute": "EC2 p3.2xlarge", "storage": "S3", "use_case": "ML"}}

Now extract from:
Text: {user_text}
Output:"""
```

**Chain-of-thought — force step-by-step reasoning:**
```python
COT_PROMPT = """
Analyze this AWS architecture for potential issues.
Think step by step:
1. Security: Are there any IAM or network security concerns?
2. Reliability: Are there single points of failure?
3. Performance: Are there bottlenecks?
4. Cost: Are there optimization opportunities?
5. Conclusion: Summarize the top 3 recommendations.

Architecture: {architecture}
"""
```

---

## 8. Error Handling Patterns

**Why required:** Production AI systems fail — throttling, timeouts, malformed outputs, empty retrievals. Robust error handling is what separates prototypes from production.

**Retry with exponential backoff:**
```python
import time
import functools
from botocore.exceptions import ClientError

def bedrock_retry(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except ClientError as e:
                    if e.response["Error"]["Code"] == "ThrottlingException":
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"Throttled. Retrying in {delay}s...")
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator

@bedrock_retry(max_retries=3)
def safe_invoke(prompt: str) -> str:
    ...
```

**Graceful RAG fallback:**
```python
def rag_with_fallback(query: str) -> str:
    try:
        docs = retriever.invoke(query)
        if not docs:
            # No relevant docs found — fall back to LLM-only
            return llm.invoke(query).content
        return rag_chain.invoke(query)
    except Exception as e:
        print(f"RAG error: {e}")
        return "I'm unable to answer that right now. Please try again."
```

**Output parsing with validation:**
```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    risk_level: str
    issues: list[str]
    recommendations: list[str]

parser = JsonOutputParser(pydantic_object=AnalysisResult)

chain = prompt | llm | parser

try:
    result = chain.invoke({"architecture": arch_description})
    print(result.risk_level)
except Exception as e:
    print(f"LLM returned invalid JSON: {e}")
    # fallback: return raw string response
```

---

## 9. Production Patterns & Best Practices

### Observability — LangSmith tracing
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
os.environ["LANGCHAIN_PROJECT"] = "bedrock-rag-prod"
# Now all LangChain calls are automatically traced
```

### Token cost tracking
```python
from langchain_community.callbacks import get_openai_callback

# For Bedrock, track manually:
response = client.converse(...)
input_cost = response["usage"]["inputTokens"] * 0.000003    # Claude 3 Sonnet pricing
output_cost = response["usage"]["outputTokens"] * 0.000015
print(f"Cost: ${input_cost + output_cost:.6f}")
```

### Secrets — never hardcode
```python
# BAD
client = boto3.client("bedrock-runtime", aws_access_key_id="AKIA...")

# GOOD — use IAM role (IRSA on EKS, instance profile on EC2)
client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
```

### Async LangChain chains
```python
# LangChain runnables support .ainvoke() and .astream()
answer = await rag_chain.ainvoke("query")

async for chunk in rag_chain.astream("query"):
    print(chunk, end="", flush=True)
```

### Document metadata filtering (Chroma)
```python
# Filter retrieval by metadata — useful for multi-tenant RAG
results = vectorstore.similarity_search(
    query="refund policy",
    k=4,
    filter={"department": "customer_service", "year": 2024}
)
```

---

## 10. Learning Roadmap

### Phase 1 — Python Foundations (Week 1–2)
- [ ] Classes, inheritance, `super()`, abstract methods
- [ ] `async/await`, `asyncio.gather()`, `asyncio.run()`
- [ ] Type hints: `TypedDict`, `Optional`, `Literal`, `Annotated`
- [ ] Decorators: understand `functools.wraps`, write a retry decorator
- [ ] Pydantic V2: `BaseModel`, `Field`, validators, `model_dump()`
- [ ] `python-dotenv`: `.env` file workflow

**Mini project:** Build a typed Python wrapper class around the boto3 Bedrock `converse` API with retry logic and config loaded from `.env`.

---

### Phase 2 — AWS Bedrock Direct (Week 3)
- [ ] Set up IAM policy and enable model access in Bedrock console
- [ ] `boto3` client: `invoke_model` and `converse`
- [ ] Handle `ClientError` exceptions (throttling, access denied)
- [ ] Implement streaming with `converse_stream`
- [ ] Multi-turn conversations with message history

**Mini project:** Build a CLI chatbot using only `boto3` + Bedrock Converse API with streaming output and conversation history.

---

### Phase 3 — LangChain (Week 4–5)
- [ ] LCEL pipe operator and Runnable interface
- [ ] `ChatBedrock` with system messages and streaming
- [ ] `ChatPromptTemplate`, `MessagesPlaceholder`
- [ ] Document loaders: PDF, directory, S3
- [ ] `RecursiveCharacterTextSplitter`
- [ ] `BedrockEmbeddings` + `Chroma` vector store
- [ ] Build complete RAG chain with LCEL

**Mini project:** RAG system over AWS documentation — ask questions, get answers with source citations.

---

### Phase 4 — LangGraph Agents (Week 6–8)
- [ ] `StateGraph`, `TypedDict` state, `add_messages` reducer
- [ ] Define nodes as Python functions
- [ ] Fixed edges and conditional routing
- [ ] `MemorySaver` checkpointer for multi-turn memory
- [ ] Build a ReAct agent with `@tool` decorated functions
- [ ] Add RAG retrieval as a tool within the agent

**Mini project:** Multi-step agent that can: (1) search your vector store, (2) call a weather API, (3) reason across multiple steps to answer a complex question.

---

### Phase 5 — Production Hardening (Week 9–10)
- [ ] LangSmith tracing and observability
- [ ] Token counting and cost tracking
- [ ] Async chains (`ainvoke`, `astream`)
- [ ] Pinecone for scalable vector storage
- [ ] Error handling: retry, fallback, output validation
- [ ] Containerize with Docker, deploy on EKS with IRSA

**Final project:** End-to-end RAG + agent system deployed on EKS, using IRSA for Bedrock authentication, Pinecone for vector storage, LangGraph for orchestration, and LangSmith for observability.

---

## Quick Reference: Packages to Install

```bash
pip install \
  boto3 \
  botocore \
  python-dotenv \
  pydantic \
  langchain-core \
  langchain-aws \
  langchain-community \
  langchain-chroma \
  langchain-text-splitters \
  langgraph \
  chromadb \
  pypdf \
  pinecone-client \
  langchain-pinecone
```

---

*Guide version: June 2026 | Covers LangChain 0.3+, LangGraph 0.2+, Pydantic V2, boto3 1.34+*
