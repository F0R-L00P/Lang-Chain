from llm_sandbox import SandboxSession

with SandboxSession(lang="python") as session:
    result = session.run(
        'print("Hello from inside the sandbox!")',
        timeout=30,
    )

print(result.stdout)