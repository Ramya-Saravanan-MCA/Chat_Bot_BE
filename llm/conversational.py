import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


def _format_chunk_id_context(top_texts, chunk_ids):
    """
    Map retrieved passages to their chunk IDs so the model can
    explicitly reference which chunk it used.
    """
    chunk_context = []
    for i, (text, chunk_id) in enumerate(zip(top_texts, chunk_ids)):
        # keep each passage relatively short; truncate if needed (optional)
        snippet = text.strip()
        if not snippet:
            continue
        # Use chunk_id if available, otherwise fall back to index
        identifier = f"[Chunk {chunk_id}]" if chunk_id is not None else f"[{i+1}]"
        chunk_context.append(f"{identifier} {snippet}")
    return "\n\n".join(chunk_context)

def get_conversational_answer(top_texts, query, model_type="openai", past_turns_context="", chunk_ids=None):
    """
    Generate a concise conversational answer strictly based on retrieved context
    and past conversation. If relevant info is not present, returns the exact
    string: "I do not have knowledge about this"
    
    Args:
        top_texts: List of text chunks
        query: User's question
        model_type: "openai" or "groq"
        past_turns_context: Summary of past conversation
        chunk_ids: List of chunk IDs corresponding to top_texts
    """
    # If chunk_ids not provided, fall back to numbered context
    if chunk_ids is None:
        chunk_ids = [None] * len(top_texts)
    
    chunk_context = _format_chunk_id_context(top_texts, chunk_ids)

    template = """
SYSTEM INSTRUCTIONS (READ FIRST):
You are a friendly, concise, and factual bank assistant. IMPORTANT — follow these rules exactly:
1) Use ONLY the text provided in the "Context" block and the "Conversation so far" block. Do not use or invent outside/world knowledge.
2) If the answer can be directly supported by Context and/or the Conversation so far, produce 2–3 short, clear, conversational sentences that directly answer the user's question.
   - When you use content from the Context, include the chunk identifiers you used in parentheses at the end of the answer, e.g. (Chunk 123) or (Chunk 45, Chunk 67).
   - Do not add long intros, apologies, or filler.
3) If the Context and Conversation do NOT contain enough information to answer the question, reply exactly (no extra punctuation, no explanation, nothing else):
I apologize, I do not have knowledge about this
4) Never hallucinate. If uncertain, use rule #3.
5) Keep style: informal, friendly, human, coversational — short 2-3 sentences only.


Conversation so far:
{past_turns}

Context:
{context}

User's question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["past_turns", "context", "question"],
        template=template,
    )

    formatted_prompt = prompt.format(
        past_turns=past_turns_context or "",
        context=chunk_context or "None",
        question=query,
    )

    # Choose deterministic / low-hallucination LLM settings
    if model_type == "openai":
        llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.0,              
            max_tokens=200,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif model_type == "groq":
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
            max_tokens=200,
        )
    else:
        raise ValueError("Invalid model type")

    # Keep the call consistent with wrapper. Here we assume .invoke() accepts a single string prompt.
    response = llm.invoke(formatted_prompt)
    # If your wrapper returns a richer object, adapt accordingly, e.g. response.content or response["text"]
    answer_text = getattr(response, "content", response)

    # final safety check: if model returned something other than exact fallback when it should have,
    # we keep it as-is;

    return answer_text.strip()