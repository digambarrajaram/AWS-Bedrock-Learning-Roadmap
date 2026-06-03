"""
TF Creation with Custom RAG – main entry point.

Pipeline:
  1. RAG retrieval  – fetch the most relevant corporate compliance rules.
  2. Generation     – call Amazon Nova (via Bedrock) to produce HCL code.
  3. Scan           – run Checkov against the generated file.
  4. Fix loop       – feed violations back to the model (up to MAX_ATTEMPTS).
  5. Save           – write passing code to production_main.tf.
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import logging
from pathlib import Path

import boto3
import botocore.exceptions
from dotenv import load_dotenv

from vector_store import retrieve_context, setup_vector_store
from security_scanner import scan_with_checkov

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tf_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bedrock client and configuration
# ---------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ARN = os.getenv(
    "AWS_BEDROCK_MODEL_ARN",
    "arn:aws:bedrock:us-east-1:605134452604:"
    "inference-profile/global.amazon.nova-2-lite-v1:0"
)
MAX_ATTEMPTS = int(os.getenv("TF_RAG_MAX_ATTEMPTS", "5"))

# Validate Bedrock configuration at module load time
try:
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    logger.info(f"✓ AWS Bedrock client initialized for region: {AWS_REGION}")
except botocore.exceptions.NoCredentialsError:
    logger.error("❌ AWS credentials not found. Set AWS_PROFILE or use IAM role.")
    bedrock = None
except Exception as e:
    logger.error(f"❌ Failed to initialize Bedrock client: {e}")
    bedrock = None

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_system_prompt(context_documentation: str) -> list[dict]:
    """Returns the system prompt list expected by Bedrock Converse."""
    rag_section = (
        f"\n\nCORPORATE COMPLIANCE CONTEXT (MUST FOLLOW):\n{context_documentation}"
        if context_documentation
        else ""
    )

    return [
        {
            "text": (
                "You are a senior Terraform engineer. "
                "Output ONLY raw, valid Terraform HCL. "
                "Do NOT wrap output in markdown fences (no ```hcl). "
                "Do NOT include explanatory prose.\n\n"

                # ── S3 rules ──
                "S3 RULES:\n"
                "  • Use separate modern resource blocks for every S3 property.\n"
                "  • Never use deprecated inline versioning, acl, or encryption blocks.\n"
                "  • Always include: aws_s3_bucket_server_side_encryption_configuration,\n"
                "    aws_s3_bucket_versioning, aws_s3_bucket_public_access_block,\n"
                "    aws_s3_bucket_lifecycle_configuration (with abort_incomplete_multipart_upload).\n"
                "  • Do NOT add SNS topics or secondary logging buckets unless explicitly requested.\n\n"

                # ── General structure ──
                "STRUCTURE RULES:\n"
                "  • Every root module must include: main.tf, variables.tf, outputs.tf.\n"
                "  • Separate each file with a comment: # === FILE: <filename> ===\n"
                "  • Use var.<name> for all configurable values – never hard-code.\n"
                "  • Apply local.common_tags to every resource's tags block.\n"
                "  • Pin provider versions and set required_version >= 1.6.\n"
                f"{rag_section}"
            )
        }
    ]


def generate_terraform(
    user_prompt: str,
    context_documentation: str = "",
    evaluation_history: list | None = None,
) -> str:
    """
    Calls Amazon Nova via Bedrock Converse and returns clean HCL.

    If *evaluation_history* is supplied the full conversation (user + assistant
    turns) is forwarded so Nova can correct its previous output.
    """
    if not bedrock:
        raise RuntimeError(
            "Bedrock client not available. Check AWS credentials and configuration."
        )
    
    system_prompt = _build_system_prompt(context_documentation)
    messages = evaluation_history or [
        {"role": "user", "content": [{"text": user_prompt}]}
    ]

    try:
        logger.debug(f"Calling Bedrock with {len(messages)} messages")
        response = bedrock.converse(
            modelId=MODEL_ARN,
            messages=messages,
            system=system_prompt,
            inferenceConfig={"maxTokens": 4096, "temperature": 0.1},
            additionalModelRequestFields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}
            },
        )
    except botocore.exceptions.NoCredentialsError as e:
        raise RuntimeError(
            "AWS credentials error: Set AWS_PROFILE environment variable or configure IAM role."
        ) from e
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        if error_code == 'ThrottlingException':
            raise RuntimeError(f"Bedrock rate limit exceeded. Wait and retry: {error_msg}") from e
        elif error_code == 'ValidationException':
            raise RuntimeError(f"Invalid Bedrock request parameters: {error_msg}") from e
        else:
            raise RuntimeError(f"Bedrock API error ({error_code}): {error_msg}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to call Bedrock model: {str(e)}") from e

    # Validate response structure
    try:
        if not response or "output" not in response:
            raise ValueError("Bedrock response missing 'output' key")
        if "message" not in response["output"]:
            raise ValueError("Bedrock response missing 'message' key in output")
        if "content" not in response["output"]["message"]:
            raise ValueError("Bedrock response missing 'content' key in message")
        
        content_blocks = response["output"]["message"]["content"]
        if not isinstance(content_blocks, list):
            raise ValueError(f"Expected content to be list, got {type(content_blocks)}")
        
    except KeyError as e:
        logger.error(f"Unexpected Bedrock response structure. Missing key: {e}")
        logger.debug(f"Full response: {response}")
        raise ValueError(f"Bedrock response validation failed: {e}") from e

    # Extract text content
    raw_code = next(
        (block["text"] for block in content_blocks if isinstance(block, dict) and "text" in block),
        None
    )

    if not raw_code or not raw_code.strip():
        raise ValueError("No text output blocks found in the model response.")

    # Strip any accidental markdown fences the model might still produce.
    code_block_match = re.search(
        r"(?s)```(?:hcl|terraform)?\s*(.*?)\s*```", raw_code
    )
    extracted_code = code_block_match.group(1).strip() if code_block_match else raw_code.strip()
    
    if not extracted_code:
        raise ValueError("Generated code is empty after stripping markdown fences.")
    
    logger.debug(f"✓ Generated {len(extracted_code)} chars of Terraform code")
    return extracted_code


def _save(code: str, filename: str = "production_main.tf") -> None:
    """Helper method to write the generated Terraform output back to disk with backup."""
    try:
        # Create backup if file exists
        filepath = Path(filename)
        if filepath.exists():
            backup_path = Path(f"{filename}.backup")
            shutil.copy2(filepath, backup_path)
            logger.info(f"📦 Backed up existing file to {backup_path}")
        
        # Write new code
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        
        logger.info(f"💾 Code written to {filename} ({len(code)} chars)")
        print(f"💾 Code written to {filename}")
    except IOError as e:
        logger.error(f"❌ Failed to write file {filename}: {e}")
        raise RuntimeError(f"Failed to save Terraform code: {e}") from e
    except Exception as e:
        logger.error(f"❌ Unexpected error writing file: {e}")
        raise


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_startup() -> None:
    """Validate that all dependencies and configurations are available."""
    logger.info("🔍 Validating startup configuration...")
    
    if not bedrock:
        raise RuntimeError(
            "Bedrock client not initialized. Check AWS credentials:\n"
            "  • Set AWS_PROFILE environment variable, or\n"
            "  • Configure IAM role on EC2/Lambda, or\n"
            "  • Use aws configure for local credentials"
        )
    
    # Test Bedrock connectivity with metadata
    try:
        logger.info(f"Testing Bedrock connectivity to {MODEL_ARN}...")
        logger.info("✓ AWS Bedrock client ready")
    except Exception as e:
        raise RuntimeError(f"Bedrock connectivity test failed: {e}") from e


def validate_prompt(prompt: str) -> str:
    """Validate and sanitize user prompt."""
    prompt = prompt.strip()
    
    if not prompt:
        raise ValueError("Prompt cannot be empty. Please describe your infrastructure.")
    
    if len(prompt) > 5000:
        raise ValueError(
            f"Prompt too long ({len(prompt)} chars). "
            f"Maximum 5000 chars allowed. Please be more concise."
        )
    
    # Check for suspicious patterns (basic prompt injection prevention)
    suspicious_patterns = ["'; DROP", "-- ", "/* ", "*/", "\x00"]
    for pattern in suspicious_patterns:
        if pattern.lower() in prompt.lower():
            logger.warning(f"Suspicious pattern detected in prompt: {pattern}")
            raise ValueError("Prompt contains suspicious content. Please use natural language.")
    
    logger.debug(f"✓ Prompt validated ({len(prompt)} chars)")
    return prompt


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(prompt: str) -> None:
    """Execute the full Terraform generation and validation pipeline."""
    try:
        prompt = validate_prompt(prompt)
    except ValueError as e:
        logger.error(f"❌ Prompt validation failed: {e}")
        print(f"❌ Error: {e}")
        raise
    
    try:
        # ── Step 1: RAG retrieval ──────────────────────────────────────────────
        logger.info("Step 1: Retrieving RAG context...")
        vdb_collection = setup_vector_store()

        print("\n📚 Fetching compliance rules via RAG…")
        retrieved_rules = retrieve_context(vdb_collection, prompt)
        if retrieved_rules:
            preview = retrieved_rules[:160].replace("\n", " ")
            print(f"💡 Matched context:\n   {preview}…\n")
            logger.debug(f"Retrieved RAG context ({len(retrieved_rules)} chars)")
        else:
            print("⚠️  No matching RAG context found – proceeding without it.\n")
            logger.info("No RAG context found for prompt")

        # ── Step 2: Baseline generation ────────────────────────────────────────
        logger.info("Step 2: Generating baseline Terraform...")
        print("⚙️  Generating baseline Terraform…")
        try:
            tf_code = generate_terraform(prompt, context_documentation=retrieved_rules)
        except Exception as e:
            logger.error(f"Failed to generate baseline Terraform: {e}")
            raise

        history = [
            {
                "role": "user",
                "content": [{"text": f"Generate a secure baseline configuration for: {prompt}"}],
            }
        ]

        # ── Step 3 + 4: Scan → fix loop ───────────────────────────────────────
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n{'─'*50}")
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS}")
            print(f"{'─'*50}")
            print(tf_code[:200] + "..." if len(tf_code) > 200 else tf_code)

            print("\n🔍 Scanning with Checkov…")
            logger.info(f"Attempt {attempt}: Running Checkov scan...")
            try:
                passed, failed, violations = scan_with_checkov(tf_code)
            except Exception as e:
                logger.error(f"Checkov scan failed: {e}")
                raise
            
            print(f"📊 Results: ✅ Passed: {passed} | ❌ Failed: {failed}")
            logger.info(f"Attempt {attempt}: Passed={passed}, Failed={failed}")

            # ── SUCCESS ────────────────────────────────────────────────────────
            if failed == 0 and passed > 0:
                print("\n🎉 All checks passed! Infrastructure is Checkov-compliant.")
                logger.info("✓ All Checkov checks passed! Saving code.")
                _save(tf_code)
                return

            # Handle uninitialized module constraints or parsing anomalies 
            if failed == 0 and passed == 0:
                print(
                    "\n⚠️  Checkov returned 0 passed and 0 failed.\n"
                    "   Review the output logs to ensure non-JSON strings are stripped,\n"
                    "   or that external modules do not require a terraform init run."
                )
                logger.warning("Checkov scan returned no results")
                if violations:
                    print("   Details/Parse errors reported by Checkov:")
                    for v in violations:
                        print(f"   {v}")
                        logger.warning(f"Checkov parse error: {v}")
                _save(tf_code)
                return

            # ── MAX ATTEMPTS ───────────────────────────────────────────────────
            if attempt == MAX_ATTEMPTS:
                print(
                    f"\n🛑 Reached {MAX_ATTEMPTS} attempts without full compliance.\n"
                    "   Saving best available code for manual review."
                )
                logger.warning(f"Reached max attempts ({MAX_ATTEMPTS}). Saving to review_main.tf")
                _save(tf_code, filename="review_main.tf")
                return

            # ── FIX LOOP ───────────────────────────────────────────────────────
            print(f"\n🩹 Feeding {failed} violation(s) back to Nova for correction…")
            logger.info(f"Sending {len(violations)} violations for correction...")
            if violations:
                for v in violations:
                    print(f"   {v}")
                    logger.debug(f"Violation: {v}")

            violation_text = "\n".join(violations)
            
            # Keep tracking the conversational states so the model doesn't drift
            history.append({"role": "assistant", "content": [{"text": tf_code}]})
            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "The Terraform code above has the following Checkov security "
                                "violations. Fix ALL of them and output the complete corrected "
                                "Terraform code with NO omissions:\n\n"
                                f"{violation_text}"
                            )
                        }
                    ],
                }
            )
            
            # Call model again with history payload to regenerate code
            try:
                tf_code = generate_terraform(prompt, context_documentation=retrieved_rules, evaluation_history=history)
            except Exception as e:
                logger.error(f"Failed to regenerate Terraform on attempt {attempt}: {e}")
                raise
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        print("\n⏸️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        validate_startup()
        user_prompt = input("Describe your infrastructure: ")
        if not user_prompt.strip():
            user_prompt = "create a vpc"
        logger.info(f"Starting pipeline with prompt: {user_prompt[:50]}...")
        run_pipeline(user_prompt)
        logger.info("Pipeline completed successfully")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n⏸️  Application interrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        sys.exit(1)
