import subprocess


def generate_answer(
    question: str,
    context_chunks: list[str],
    model_name: str = "llama3:8b",
) -> str:
    """
    Generate an answer using Ollama and retrieved context.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a scientific assistant.

Answer the question using only the context provided below.
If the answer is not contained in the context, say that the available context is insufficient.

Context:
{context}

Question:
{question}

Answer:
"""

    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        text=True,
        capture_output=True,
        check=True,
    )

    return result.stdout.strip()