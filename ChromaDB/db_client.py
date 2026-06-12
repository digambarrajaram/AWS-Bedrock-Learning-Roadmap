import chromadb


chroma_client = chromadb.Client()


collection = chroma_client.create_collection(name="my_collection")


collection.add(
    ids=["id1", "id2", "id3", "id4"],
    documents=[
        "Apple is leading in a smart phone game with iPhone sales up by 35%",
        "Tesla booked a minor profit of 1 billion $ in Q2",
        "Apples are high in fiber, vitamin C, and various antioxidants",
        "SpaceX got NASA contract worth 10 billion $"
    ]
)


results = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

# 2. Get the total number of items stored
total_items = len(results["ids"])

# 3. Loop through using the index to print them side-by-side
for i in range(total_items):
    print(f"--- ITEM {i+1} ---")
    print(f"ID: {results['ids'][i]}")
    print(f"Document: {results['documents'][i]}")
    print(f"Metadata: {results['metadatas'][i]}")
    
    # Grab the 768-dimension vector array
    vector = results['embeddings'][i]
    # Print just the first 3 numbers so it doesn't flood your screen
    print(f"Embedding (First 3 dims of {len(vector)} total): {vector[:3]}...") 
    print("\n")


# 1. You must add include=["documents", "metadatas", "embeddings", "distances"]
results = collection.query(
    query_texts=["who is Elon Musk?"],
    n_results=2,
    include=["documents", "metadatas", "embeddings", "distances"] 
)

# 2. Extract the lists for the FIRST query text (index 0)
# Chroma returns a list-of-lists because you can pass multiple query_texts at once
ids = results["ids"][0]
documents = results["documents"][0]
metadatas = results["metadatas"][0]
embeddings = results["embeddings"][0]
distances = results["distances"][0]

print("\n\n")
# 3. Loop through the results safely
for i in range(len(ids)):
    print(f"ITEM: {i+1}")
    print("ID:", ids[i])
    print("Document:", documents[i])
    print("Metadata:", metadatas[i])
    print("Distance Score:", distances[i]) # Lower means closer/more relevant
    
    # Slice the embedding array so it doesn't flood your console terminal
    print("Embedding (First 3 dimensions):", embeddings[i][:3], "...")
    print("\n")

