import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_api.modules.prompt_builder import build_prompt
from rag_api.modules.retrieval import get_top_k_chunks

MODEL_NAME = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def query_llm(prompt: str, max_tokens: int = 300) -> str:
    """
    Generates an answer from a free Hugging Face model.
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    answer = output_text[len(prompt) :].strip()
    return answer


def main():

    query = input("Enter your query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    sorted_chunks = get_top_k_chunks(query)

    text_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]

    prompt = build_prompt(query, text_chunks)

    answer = query_llm(prompt)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    for i, chunk in enumerate(text_chunks, start=1):
        print(f"{i}. {chunk['source_url']}")


if __name__ == "__main__":
    main()
