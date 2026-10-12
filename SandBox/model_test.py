from transformers import pipeline
from llm_sandbox import SandboxSession

model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

generator = pipeline(
    "text-generation",
    model=model_id,
    device=0,
    dtype="auto",
)

messages = [
    {
        "role": "user",
        "content": "Write Python code that prints the squares of numbers 1 through 5. Return only Python code.",
    }
]

response = generator(
    messages,
    max_new_tokens=100,
    do_sample=False,  # makes the response more predictable, which is useful for code.
)

generated_code = response[0]["generated_text"][-1]["content"]
print("Model generated:\n", generated_code)

generated_code = generated_code.strip()

if generated_code.startswith("```") and generated_code.endswith("```"):
    lines = generated_code.splitlines()
    generated_code = "\n".join(lines[1:-1])

print("Cleaned code:\n", generated_code)

sandbox_limits = {
    # Docker creates the container without a network interface. Generated code cannot access websites, APIs, or devices on your local network.
    "network_mode": "none",
    # The container can consume at most 512 MB of RAM
    "mem_limit": "512m",
    # The container can have at most 128 processes
    "pids_limit": 128,
    # Processes inside the container cannot gain additional operating-system privileges.
    "security_opt": ["no-new-privileges:true"],
}

with SandboxSession(
    lang="python",
    runtime_configs=sandbox_limits,
) as session:
    result = session.run(generated_code, timeout=30)

print("\nSandbox output:")
print(result.stdout)

if result.stderr:
    print("Sandbox errors:")
    print(result.stderr)
