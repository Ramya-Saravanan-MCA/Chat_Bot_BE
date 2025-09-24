from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os

# Load LLM and prompt template only once
summarizer_prompt = """
You are a conversation summarizer with a focus on producing clear and concise rolling summaries.

Rules:
- Output only the updated summary with no explanations or extra text.
- Limit to two or three crisp sentences with maximum brevity.
- Always merge the previous summary with the latest turn into one coherent summary.
- Preserve key details with intent and context while removing repetition and filler.
- Prioritize clarity with readability at all times.

Previous summary:
{past_summary}

Latest turn:
{last_turn}

Updated summary : """


sum_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
sum_prompt = PromptTemplate.from_template(summarizer_prompt)
_chain = sum_prompt | sum_llm

def summarizer(last_turn: str, past_summary: str = "") -> str:
    response = _chain.invoke({
        "past_summary": past_summary or "",
        "last_turn": last_turn
    })
    return response.content.strip()