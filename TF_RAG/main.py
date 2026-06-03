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

import re
import sys

import boto3

from vector_store import retrieve_context, setup_vector_store
from security_scanner import scan_with_checkov

# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL_ARN = (
    "arn:aws:bedrock:us-east-1:605134452604:"
    "inference-profile/global.amazon.nova-2-lite-v1:0"
)

MAX_ATTEMPTS = 5

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
    system_prompt = _build_system_prompt(context_documentation)
    messages = evaluation_history or [
        {"role": "user", "content": [{"text": user_prompt}]}
    ]

    response = bedrock.converse(
        modelId=MODEL_ARN,
        messages=messages,
        system=system_prompt,
        inferenceConfig={"maxTokens": 4096, "temperature": 0.1},
        additionalModelRequestFields={
            "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}
        },
    )

    content_blocks = response["output"]["message"]["content"]
    raw_code = next(
        (block["text"] for block in content_blocks if "text" in block), ""
    )

    if not raw_code:
        raise ValueError("No text output blocks found in the model response.")

    # Strip any accidental markdown fences the model might still produce.
    code_block_match = re.search(
        r"(?s)```(?:hcl|terraform)?\s*(.*?)\s*```", raw_code
    )
    return code_block_match.group(1).strip() if code_block_match else raw_code.strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(prompt: str) -> None:
    # ── Step 1: RAG retrieval ──────────────────────────────────────────────
    vdb_collection = setup_vector_store()

    print("\n📚 Fetching compliance rules via RAG…")
    retrieved_rules = retrieve_context(vdb_collection, prompt)
    if retrieved_rules:
        preview = retrieved_rules[:160].replace("\n", " ")
        print(f"💡 Matched context:\n   {preview}…\n")
    else:
        print("⚠️  No matching RAG context found – proceeding without it.\n")

    # ── Step 2: Baseline generation ────────────────────────────────────────
    print("⚙️  Generating baseline Terraform…")
    tf_code = generate_terraform(prompt, context_documentation=retrieved_rules)

    # ── Step 3 + 4: Scan → fix loop ───────────────────────────────────────
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'─'*50}")
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS}")
        print(f"{'─'*50}")
        print(tf_code)

        print("\n🔍 Scanning with Checkov…")
        passed, failed, violations = scan_with_checkov(tf_code)
        print(f"📊 Results: ✅ Passed: {passed} | ❌ Failed: {failed}")

        # ── SUCCESS ────────────────────────────────────────────────────────
        # Guard: only treat as success when Checkov actually evaluated checks.
        # passed=0 AND failed=0 usually means Checkov couldn't scan the file
        # (e.g. module-only code without terraform init, or a parse error).
        if failed == 0 and passed > 0:
            print("\n🎉 All checks passed! Infrastructure is Checkov-compliant.")
            _save(tf_code)
            return

        if failed == 0 and passed == 0:
            print(
                "\n⚠️  Checkov returned 0 passed and 0 failed.\n"
                "   This usually means the generated code uses external modules\n"
                "   that require 'terraform init' before Checkov can lint them,\n"
                "   or Checkov could not parse the file.\n"
                "   The code has been saved for manual review."
            )
            if violations:
                print("   Parse errors reported by Checkov:")
                for v in violations:
                    print(f"   {v}")
            _save(tf_code)
            return

        # ── MAX ATTEMPTS ───────────────────────────────────────────────────
        if attempt == MAX_ATTEMPTS:
            print(
                f"\n🛑 Reached {MAX_ATTEMPTS} attempts without full compliance.\n"
                "   Saving best available code for manual review."
            )
            _save(tf_code, filename="review_main.tf")
            return

        # ── FIX LOOP ───────────────────────────────────────────────────────
        print(f"\n🩹 Feeding {failed} violation(s) back to Nova for correction…")
        if violations:
            for v in violations:
                print(f"   {v}")

        violation_text = "\n".join(violations)
        history = [
            {
                "role": "user",
                "content": [
                    {"text": f"Generate a secure baseline configuration for: {prompt}"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"text": tf_code}],
            },
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
            },
        ]

        tf_code = generate_terraform(
            prompt,
            context_documentation=retrieved_rules,
            evaluation_history=history,
        )


def _save(tf_code: str, filename: str = "production_main.tf") -> None:
    with open(filename, "w") as f:
        f.write(tf_code)
    print(f"💾 Code written to {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow passing prompt as CLI arg for scripted use.
        user_prompt = " ".join(sys.argv[1:])
    else:
        user_prompt = input("Describe your infrastructure: ").strip()

    if not user_prompt:
        print("❌ No prompt provided. Exiting.")
        sys.exit(1)

    run_pipeline(user_prompt)