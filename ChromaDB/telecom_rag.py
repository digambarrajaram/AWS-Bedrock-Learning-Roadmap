from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
load_dotenv()


#Step 1: PDF or any documentation loading. knowledge base for RAG
pdf_path = "telecom_guide.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print("-----------------printing first page content--------------------")
print(pages[1].page_content)




#Step 2: chunking step for provided knowledge base for RAG
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")
print("first chunk : ",chunks[0].page_content)
print("second chunk : ",chunks[1].page_content)


#Step 3: Embeddings for knowledge base for RAG
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Corrected syntax for standard LangChain usage
vectorstore = Chroma.from_documents(
    documents=chunks,              # Changed 'chunks' to 'documents'
    embedding=embeddings,           # Changed 'embeddings' to 'embedding'
    persist_directory="./chroma_db"
)

print("Embeddings created")
print(embeddings)
print(vectorstore)


# 3. Perform a similarity search
query = "What is VoLTE and how does it work?"
retriver = vectorstore.as_retriever(search_kwargs={"k": 3})

result = retriver.invoke(query)
print("\n\n")
# 4. Print the retrieved text chunks
for i, doc in enumerate(result, 1):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content)


# --- Helper: join retrieved chunks into a single context string ---
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# --- System prompt: ground the LLM in the retrieved context ---
SYSTEM_PROMPT = """\
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# --- LLM via Groq API ---
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed",
)

# --- Assemble the chain ---
chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain assembled.")


question = "How does international roaming work and what charges should I expect?"

print("\n\n")
print(f"Q: {question}\n")
print("A:", chain.invoke(question))


print("\n\n")

print("Telecom RAG Assistant — type 'quit' to exit\n")

while True:
    question = input("Your question: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not question:
        continue

    print("\nAnswer:")
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")



debug_question = "What security measures protect against SIM swap fraud?"

docs = retriver.invoke(debug_question)
print(f"Question: {debug_question}")
print(f"Retrieved {len(docs)} chunks:\n")
for i, doc in enumerate(docs, 1):
    print(f"{'='*60}")
    print(f"Chunk {i} (page {doc.metadata.get('page', '?')})")
    print(f"{'='*60}")
    print(doc.page_content)
    print()

print("\nFinal Answer:")
print(chain.invoke(debug_question))