"""Reader: reads documentation text and produces a structured task spec.

This is step 1 of the loop. The reader does ONE job here: given some docs text
and a topic, distill a precise spec describing a function for the coder to build.
(Test-case generation will be a separate call, added later.)

Model id is a constant so we can swap 1.5B -> 7B-Instruct without touching logic.
"""

import json
import re

from transformers import pipeline

# Reusing the already-downloaded model so iteration is fast.
# Swap to "Qwen/Qwen2.5-7B-Instruct" once the flow is proven.
READER_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Loaded once at import so repeated calls don't reload the model.
_reader = pipeline(
    "text-generation",
    model=READER_MODEL_ID,
    device=0,
    dtype="auto",
)

SPEC_SYSTEM_PROMPT = """You are a precise technical reader.
Given documentation text and a topic, you produce a specification for a single
Python function that a programmer will implement. You do NOT write the function.
You do NOT write tests. You only describe what the function must do.

Respond with ONLY a JSON object, no prose, no markdown fences, in this shape:
{
  "function_name": "snake_case_name",
  "signature": "function_name(arg: type) -> return_type",
  "description": "one or two sentences on what the function does",
  "notes": ["edge case or detail 1", "detail 2"]
}"""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Weak models often wrap JSON in prose or ```json fences, so we find the
    outermost { ... } rather than trusting the whole string to be clean JSON.
    """
    text = text.strip()

    # Strip a leading/trailing markdown fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Grab from the first { to the last } — the spec object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    candidate = text[start : end + 1]
    return json.loads(candidate)


def spec_from_docs(docs_text: str, topic: str) -> dict:
    """Produce a task spec (as a dict) from docs text, focused on a topic."""
    user_prompt = (
        f"Topic to focus on: {topic}\n\n"
        f"Documentation:\n{docs_text}\n\n"
        "Produce the JSON spec now."
    )

    messages = [
        {"role": "system", "content": SPEC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = _reader(
        messages,
        max_new_tokens=400,
        do_sample=False,  # deterministic for a structured task
    )
    raw = response[0]["generated_text"][-1]["content"]
    return _extract_json(raw)


if __name__ == "__main__":
    # A simple example to test the reader in isolation.
    example_docs = """
    String formatting: a slug is a short label for something, containing only
    letters, numbers, and hyphens. To make a slug from a title, lowercase the
    text, strip leading and trailing whitespace, and replace runs of whitespace
    with a single hyphen. Slugs are commonly used in URLs.
    """

    spec = spec_from_docs(example_docs, topic="creating a URL slug")

    print("Spec (parsed):")
    print(json.dumps(spec, indent=2))
