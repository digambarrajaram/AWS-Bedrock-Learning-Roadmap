import boto3
import json
import subprocess
import re


bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def generate_terraform(user_prompt: str, evaluation_history: list = None,) -> str:

    system_prompt = [{
        "text": (
            "You are a Terraform expert. Output ONLY raw, valid Terraform HCL code configurations. "
            "Do NOT wrap your output in markdown code blocks like ```hcl or include conversational text. "
            "CRITICAL: Always use separate modern resources for S3 properties. Do NOT use inline blocks for versioning, acl, or encryption. "
            "Use 'aws_s3_bucket_server_side_encryption_configuration', 'aws_s3_bucket_versioning', and 'aws_s3_bucket_public_access_block'. "
            "Ensure 'abort_incomplete_multipart_upload' is configured in any lifecycle rules. "
            "Do NOT add SNS topics or secondary logging buckets unless the user explicitly requested them."
        )
    }]
    
    # Use clean, un-inflated messages to save token space
    messages = evaluation_history if evaluation_history else [{"role": "user", "content": [{"text": user_prompt}]}]
    
    response = bedrock.converse(
        modelId="arn:aws:bedrock:us-east-1:605134452604:inference-profile/global.amazon.nova-2-lite-v1:0",
        messages=messages,
        system=system_prompt,
        inferenceConfig={"maxTokens": 4096, "temperature": 0.1},
        additionalModelRequestFields={"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}},
        performanceConfig={"latency": "standard"}
    )
    
    content_blocks = response["output"]["message"]["content"]
    raw_code = next((block["text"] for block in content_blocks if "text" in block), "")
            
    if not raw_code:
        raise ValueError("No text output blocks found in the model response.")

    cleaned_code = re.sub(r"```(?:hcl|terraform)?\s*", "", raw_code).replace("```", "").strip()
    return cleaned_code

def scan_with_checkov(tf_code: str) -> tuple:
    with open("generated.tf", "w") as f:
        f.write(tf_code)
        
    # Skip organizational level constraints (logging and replication) to keep baseline items lightweight
    result = subprocess.run(
        ["checkov", "-f", "generated.tf", "--output", "json", "--skip-check", "CKV_AWS_18,CKV_AWS_144"],
        capture_output=True, text=True, shell=True
    )
    
    passed, failed, error_messages = 0, 0, []
    
    if result.stdout and result.stdout.strip():
        try:
            scan_result = json.loads(result.stdout)
            reports = scan_result if isinstance(scan_result, list) else [scan_result]
            for report in reports:
                summary = report.get("summary", {}) or {}
                passed += summary.get("passed", 0)
                failed += summary.get("failed", 0)
                
                failed_checks = report.get("results", {}).get("failed_checks", []) or []
                for check in failed_checks:
                    error_messages.append(f"- [{check.get('check_id')}]: {check.get('check_name')}")
        except json.JSONDecodeError:
            error_messages.append("Failed to parse checkov output schema.")
            
    return passed, failed, error_messages

if __name__ == "__main__":
    prompt = input("Describe your infrastructure: ")
    print("\n⚙️  Generating Baseline Terraform...")
    
    tf_code = generate_terraform(prompt)
    MAX_ATTEMPTS = 5
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Code Instance (Attempt {attempt}/{MAX_ATTEMPTS}) ---")
        print(tf_code)
        print("------------------------------------------")
        
        print("🔍 Scanning for security violations...")
        passed, failed, violations = scan_with_checkov(tf_code)
        print(f"📊 Results: ✅ Passed: {passed} | ❌ Failed: {failed}")
        
        if failed == 0:
            print("\n🎉 Success! Infrastructure is verified secure by Checkov.")
            with open("production_main.tf", "w") as prod_file:
                prod_file.write(tf_code)
            print("💾 Production infrastructure written to production_main.tf")
            break
            
        if attempt == MAX_ATTEMPTS:
            print("\n🛑 Maximum recovery attempts reached. Manual optimization required.")
            break
            
        print("\n🩹 Fix loop initiated! Feeding security flaws back to Amazon Nova...")
        
        # SLIDING WINDOW BUFFER: Completely refresh history every loop to prevent token truncation
        history = [
            {"role": "user", "content": [{"text": f"Generate a secure baseline configuration for: {prompt}"}]},
            {"role": "assistant", "content": [{"text": tf_code}]},
            {"role": "user", "content": [{"text": "Fix the following Checkov errors in the code above and output the corrected version:\n" + "\n".join(violations)}]}
        ]
        
        tf_code = generate_terraform(prompt, evaluation_history=history)
