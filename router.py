import re
import os
import json
import time
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# --- Fast rule-based routing patterns ---
RE_HANDOFF = re.compile(r"\b(human|agent|advisor|call|book|schedule|meeting|appointment|speak|talk)\b", re.I)

def rule_gate(query):
    """Apply deterministic rules to quickly route obvious intents"""
    if RE_HANDOFF.search(query):
        return {"labels": ["handoff_request"], "slots": {}, "source": "rule"}
    return None

# --- LLM-based intent classification ---
ROUTER_SYSTEM = """You are a message router for a banking chatbot.
Return STRICT JSON with keys: labels (array), confidence (0.0-1.0), slots (object), safety_flags (array).
Valid labels: greeting, in_scope_knowledge, out_of_scope, chitchat_smalltalk, handoff_request, safety_risky.
- greeting: salutations, thanks, farewells.
- in_scope_knowledge: banking products, account info, loans, deposits, investments, credit cards, fees, interest rates, transfers, payments.
- out_of_scope: unrelated to banking/finance domain.
- chitchat_smalltalk: casual talk without a factual ask.
- handoff_request: wants a human banker, agent or meeting.
- safety_risky: abusive, sensitive/regulated request, or requests for personalized financial advice.
Extract slots when present: account_type, product_type, loan_type, card_type, transaction_type, amount, currency, time_period.
"""

ROUTER_USER_TMPL = 'Query: "{q}"\nReturn JSON only.'

def call_llm_router(query, model_type="groq"):
    """Use LLM to classify intent with structured output"""
    if model_type == "openai":
        llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
        )
    
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": ROUTER_USER_TMPL.format(q=query)}
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Handle potential JSON formatting issues
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
            
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"Error parsing router response: {e}")
        # Fallback to safe default
        return {"labels": ["in_scope_knowledge"], "confidence": 0.5, "slots": {}, "safety_flags": []}

# --- Retrieval strength calculation ---
def retrieval_strength(topk):
    """Calculate retrieval confidence from similarity scores"""
    if topk is None:
        return 0.0
    
    if hasattr(topk, 'empty') and topk.empty:
        return 0.0
        
    # Get scores from results dataframe
    if hasattr(topk, 'columns') and "final_score" in topk.columns:
        scores = topk["final_score"].tolist()
        if not scores or all(s is None for s in scores):
            return 0.0
        scores = [s for s in scores if s is not None]
        if not scores:
            return 0.0
    else:
        return 0.0
        
    # Use top 3 scores
    top_scores = sorted(scores, reverse=True)[:3]
    print(top_scores)  #for debugging or tweeking the thresholds 
    return (top_scores, sum(top_scores) / len(top_scores) if top_scores else 0.0)

# --- Threshold constants ---
TAU_RETRIEVE_STRONG = 0.030   # confident we have good context
TAU_RETRIEVE_WEAK = 0.015    # below this, likely no answer in KB
TAU_ROUTER_HIGH = 0.80      # LLM router confident

# --- Response composers ---
def reply_greeting():
    return "Hello! I'm your banking assistant. How can I help you with accounts, loans, credit cards, or other banking services today?"

def reply_handoff():
    return "I'd be happy to connect you with a banking representative. Would you prefer a phone call or an in-branch appointment? Please let me know your preferred time."

def reply_safety():
    return "I apologize, but I'm not able to provide personalized financial advice or handle this sensitive request. For your security, please contact our customer service directly at our official helpline."

def reply_oos():
    return "I'm specifically trained to help with banking and financial services queries. Is there something I can assist you with regarding accounts, loans, investments, or other banking products?"

def reply_not_found():
    """Simple, natural response when information is not found"""
    return "I don't have information about that. Could you ask about something else I can help you with?"

def reply_chitchat():
    return "I appreciate the conversation! I'm here primarily to help with banking questions. Is there anything specific about your accounts, transactions, or banking services I can assist with?"

# --- Core routing decision function ---
def route_and_answer(query, past_summary, retriever, llm_model, answer_llm, 
                    retrieval_mode="hybrid", k_dense=10, k_sparse=10, 
                    rrf_k=60, top_k_final=10, doc_filter=None):
    """
    Main routing function that decides how to handle the query
    """
    
    # Add timing tracking to router
    timing_info = {}
    
    # Check rule-based patterns first (fast path)
    rule_start = time.perf_counter() 
    ruled = rule_gate(query)
    timing_info["rule_routing_time"] = time.perf_counter() - rule_start
    
    if ruled and "handoff_request" in ruled["labels"]:
        timing_info["llm_routing_time"] = 0.0 
        return {
            "answer": reply_handoff(),
            "intent": ruled,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "handoff" 
        }
    
    # LLM router for intent classification
    llm_start = time.perf_counter() 
    intent_info = call_llm_router(query, llm_model)
    timing_info["llm_routing_time"] = time.perf_counter() - llm_start
    
    labels = set(intent_info.get("labels", []))
    confidence = float(intent_info.get("confidence", 0))
    slots = intent_info.get("slots", {})
    safety_flags = intent_info.get("safety_flags", [])
    
    # Check for safety concerns
    if "safety_risky" in labels or safety_flags:
        return {
            "answer": reply_safety(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info ,
            "answer_decision": "safety"
        }
    
    # Check for greeting-only intent
    if labels == {"greeting"}:
        return {
            "answer": reply_greeting(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info ,
             "answer_decision": "greeting"
        }
    
    # Check for chitchat/smalltalk with high confidence
    if "chitchat_smalltalk" in labels and confidence >= TAU_ROUTER_HIGH and "in_scope_knowledge" not in labels:
        return {
            "answer": reply_chitchat(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "chitchat"
        }
    
    # Check for handoff request with high confidence
    if "handoff_request" in labels and confidence >= TAU_ROUTER_HIGH:
        return {
            "answer": reply_handoff(),
            "intent": intent_info,
            "retrieval_results": None,
            "retrieval_strength": 0.0,
            "used_rag": False,
            "timing_info": timing_info,
            "answer_decision": "handoff"  
        }
    
    # Perform retrieval with user-selected parameters and STRICT doc filtering
    retrieval_start = time.perf_counter()  
    results, retrieval_timings = retriever.retrieve(
        query, 
        mode=retrieval_mode,
        k_dense=k_dense,
        k_sparse=k_sparse,
        rrf_k=rrf_k,
        top_k_final=top_k_final,
        doc_filter=doc_filter  # THIS IS KEY - strictly filter by selected documents
    )

    strength = retrieval_strength(results)[1]
    top_scores = retrieval_strength(results)[0]
    
    
    # In-scope knowledge with good retrieval
    if "in_scope_knowledge" in labels and strength >= TAU_RETRIEVE_STRONG:
        generation_start = time.perf_counter()  
        if answer_llm:
            answer = answer_llm(
                list(results["text"]),
                query,
                llm_model.lower(),
                past_summary,
                chunk_ids=list(results["chunk_id"])
            )
        else:
            # Fallback if no answer_llm provided
            answer = "I found relevant information about your banking query, but I need to check it for accuracy. Could you please provide more specific details?"
        
        timing_info["generation_time"] = time.perf_counter() - generation_start
        
        # Add greeting prefix if also greeting intent
        if "greeting" in labels:
            answer = f"Hello! {answer}"
        
        return {
            "answer": answer,
            "intent": intent_info,
            "retrieval_results": results,
            "retrieval_strength": strength,
            "used_rag": True,
            "timing_info": timing_info,  
            "retrieval_timings": retrieval_timings,
            "answer_decision": "inscope" ,
            "top_scores": top_scores,     
        }
    
    # Strong retrieval but unclear intent - still use RAG
    if strength >= TAU_RETRIEVE_STRONG:
        generation_start = time.perf_counter()  
        if answer_llm:
            answer = answer_llm(
                list(results["text"]),
                query,
                llm_model.lower(),
                past_summary,
                chunk_ids=list(results["chunk_id"])
            )
        else:
            answer = "Based on our banking information, I found something that might help, but I'd need more specific details about your banking query."
        
        timing_info["generation_time"] = time.perf_counter() - generation_start
        
        # Add greeting prefix if also greeting intent
        if "greeting" in labels:
            answer = f"Hello! {answer}"
        
        return {
            "answer": answer,
            "intent": intent_info,
            "retrieval_results": results,
            "retrieval_strength": strength,
            "used_rag": True,
            "timing_info": timing_info,
            "retrieval_timings": retrieval_timings  ,
            "answer_decision": "inscope",
            "top_scores": top_scores,    
        }
    
    # Out of scope with weak retrieval
    if "out_of_scope" in labels and strength < TAU_RETRIEVE_WEAK:
        timing_info["generation_time"] = 0.0  
        return {
            "answer": reply_oos(),
            "intent": intent_info,
            "retrieval_results": results,
            "retrieval_strength": strength,
            "used_rag": False,
            "timing_info": timing_info,  
            "retrieval_timings": retrieval_timings ,
            "answer_decision": "oos",
            "top_scores": top_scores,      
        }
    
    # Default: simple "I don't know" response
    timing_info["generation_time"] = 0.0
    return {
        "answer": reply_not_found(),  # Simple, natural response
        "intent": intent_info,
        "retrieval_results": results,
        "retrieval_strength": strength,
        "used_rag": False,
        "timing_info": timing_info, 
        "retrieval_timings": retrieval_timings,
        "answer_decision": "reply_not_found" ,
        "top_scores": top_scores,       
    }