import asyncio
import time

# 1. A simulated asynchronous function to call AWS Bedrock
async def call_bedrock_async(user_id: str, prompt: str) -> str:
    print(f"[Start] Sending query for {user_id} to AWS Bedrock...")
    
    # Instead of time.sleep() which blocks everything, we use asyncio.sleep()
    # This simulates waiting for a network API response from Claude
    await asyncio.sleep(2.0) 
    
    print(f"[Done] Received response for {user_id}!")
    return f"Response to '{prompt}'"

# 2. The main orchestrator function
async def main():
    start_time = time.time()
    
    # We launch 3 separate AI tasks simultaneously
    task1 = call_bedrock_async("User_A", "Explain Python OOP.")
    task2 = call_bedrock_async("User_B", "What is RAG?")
    task3 = call_bedrock_async("User_C", "Write a decorator example.")
    
    # asyncio.gather runs all tasks concurrently and waits for all to finish
    results = await asyncio.gather(task1, task2, task3)
    
    end_time = time.time()
    print(f"\nAll responses received in {end_time - start_time:.2f} seconds!")
    print("Results summary:", results)

# 3. How to execute an async program in Python
asyncio.run(main())
