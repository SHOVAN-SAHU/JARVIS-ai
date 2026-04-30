"""
core/agent.py
The JARVIS brain — LangChain ReAct agent powered by Groq LLaMA 3.3.

Retry system:
  - Up to MAX_RETRIES attempts per query
  - Exponential backoff between retries (1s, 2s, 4s...)
  - Different retry strategies per failure type:
      * Rate limit  → wait longer, same query
      * Parse error → rephrase the query for the LLM
      * Tool error  → try without tools (pure LLM fallback)
      * Network     → wait and retry
  - JARVIS speaks a retry acknowledgement so the user knows it's still working
"""
import logging
import time
import random
from typing import Callable

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import RateLimitError, APIConnectionError, APIStatusError

from config import settings
from tools import get_all_tools

logger = logging.getLogger(__name__)

# ── Retry Config ──────────────────────────────────────────────────────────────
MAX_RETRIES = 3
BASE_BACKOFF = 1.0      # seconds — doubles each retry (1, 2, 4)
RATE_LIMIT_WAIT = 8.0   # extra wait on 429 rate limit errors

# ── Retry spoken phrases — so user knows JARVIS is still on it ───────────────
RETRY_PHRASES = [
    "One moment, I'm having a bit of trouble. Let me try that again.",
    "Running into a snag — retrying now.",
    "Give me just a second, I'm trying a different approach.",
    "Encountered an issue. Attempting again, sir.",
]

FALLBACK_PHRASES = [
    "I couldn't complete that through my usual methods, but let me answer from what I know.",
    "My tools aren't cooperating right now, so I'll answer directly.",
]

# ── System Prompt ─────────────────────────────────────────────────────────────
JARVIS_SYSTEM_PROMPT = """You are JARVIS, an advanced AI assistant — witty, helpful, and efficient.
You were built to assist your user with intelligence and precision.

Personality:
- Speak naturally and conversationally — not like a robot
- Be concise in voice responses (they are spoken aloud)
- Occasionally show personality with dry wit, but stay professional
- When starting a task, briefly confirm what you're doing (e.g., "Searching for that now, sir.")
- When done, summarize the result clearly

You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: your final conversational response (keep it natural, it will be spoken aloud)

Important rules:
- Always end with "Final Answer:"
- Keep Final Answer responses concise and natural for voice output
- For complex results, summarize the key points rather than reading everything verbatim
- Never start your Final Answer with "Based on the search results" — just give the answer naturally

Begin!

{chat_history}
Question: {input}
Thought: {agent_scratchpad}"""

# Simpler prompt for direct LLM fallback (no tools)
FALLBACK_PROMPT = """You are JARVIS, a helpful AI assistant. Answer the following question 
conversationally and concisely — your response will be spoken aloud.

Question: {input}
Answer:"""


# ── Failure classification ────────────────────────────────────────────────────

def _classify_error(e: Exception) -> str:
    """
    Returns a string tag for the type of failure so we can
    apply the right retry strategy.
    """
    msg = str(e).lower()
    if isinstance(e, RateLimitError) or "rate limit" in msg or "429" in msg:
        return "rate_limit"
    if isinstance(e, APIConnectionError) or "connection" in msg or "timeout" in msg:
        return "network"
    if "parsing" in msg or "output" in msg or "format" in msg or "could not parse" in msg:
        return "parse"
    if isinstance(e, APIStatusError):
        return "api_error"
    return "unknown"


class JarvisAgent:
    """
    LangChain ReAct agent with autonomous retry logic.

    On failure, JARVIS:
      1. Classifies the error type
      2. Speaks a retry acknowledgement
      3. Waits (exponential backoff)
      4. Adjusts strategy (rephrase / fallback / wait longer)
      5. Tries again — up to MAX_RETRIES times
      6. Falls back to direct LLM answer if all retries fail
    """

    def __init__(self, speak_callback: Callable[[str], None] | None = None):
        self.speak_callback = speak_callback
        self.tools = get_all_tools()
        self._llm = None
        self._build_agent()

    def _build_agent(self):
        """Construct the LangChain agent executor."""
        self._llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_LLM_MODEL,
            temperature=0.6,
            max_tokens=1024,
        )

        prompt = PromptTemplate.from_template(JARVIS_SYSTEM_PROMPT)

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=False,
        )

        agent = create_react_agent(
            llm=self._llm,
            tools=self.tools,
            prompt=prompt,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            verbose=settings.AGENT_VERBOSE,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )

        logger.info(
            f"🧠  Agent ready | Model: {settings.GROQ_LLM_MODEL} | "
            f"Tools: {[t.name for t in self.tools]}"
        )

    # ── Pre-action speech ─────────────────────────────────────────────────────

    def _maybe_speak_action(self, query: str):
        """Speak a heads-up phrase before likely tool use."""
        if not self.speak_callback:
            return
        q = query.lower()
        hints = {
            ("search", "look up", "find", "google", "browse", "news"): "Alright, searching the web for that.",
            ("time", "date", "day", "today", "clock"):                  "Let me check that for you.",
            ("cpu", "battery", "memory", "ram", "disk", "system"):      "Checking your system status now.",
        }
        for keywords, phrase in hints.items():
            if any(k in q for k in keywords):
                self.speak_callback(phrase)
                return

    # ── Fallback: answer without tools ────────────────────────────────────────

    def _llm_fallback(self, query: str) -> str:
        """
        When all retries fail, ask the LLM directly without any tools.
        This almost never fails since it's just a text completion.
        """
        logger.info("🔄  Attempting direct LLM fallback (no tools)...")
        if self.speak_callback:
            self.speak_callback(random.choice(FALLBACK_PHRASES))
        try:
            from langchain_core.messages import HumanMessage
            response = self._llm.invoke([HumanMessage(content=query)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM fallback also failed: {e}")
            return (
                "I'm sorry sir, I'm experiencing technical difficulties right now. "
                "All retry attempts have been exhausted. Please try again in a moment."
            )

    # ── Retry rephrase ────────────────────────────────────────────────────────

    def _rephrase(self, query: str, attempt: int) -> str:
        """
        On parse errors, slightly rephrase the query to help
        the LLM produce a better-formatted ReAct output.
        """
        prefixes = [
            "Please answer this clearly: ",
            "In simple terms, ",
            "Step by step, ",
        ]
        idx = (attempt - 1) % len(prefixes)
        return prefixes[idx] + query

    # ── Core think() with retry loop ─────────────────────────────────────────

    def think(self, user_input: str) -> str:
        """
        Process user input with automatic retry on failure.

        Retry matrix:
          rate_limit  → sleep RATE_LIMIT_WAIT + backoff, same query
          network     → sleep backoff, same query
          parse       → sleep backoff, rephrased query
          api_error   → sleep backoff * 2, same query
          unknown     → sleep backoff, same query
        After MAX_RETRIES failures → LLM direct fallback
        """
        logger.info(f"🤔  Thinking about: \"{user_input}\"")
        self._maybe_speak_action(user_input)

        current_query = user_input
        last_error = None

        for attempt in range(1, MAX_RETRIES + 2):  # +2 so attempt 1 = first try
            is_retry = attempt > 1

            if is_retry:
                # Classify previous error and adjust strategy
                error_type = _classify_error(last_error)
                logger.warning(f"⚠️  Retry {attempt - 1}/{MAX_RETRIES} | Error type: {error_type}")

                # Speak retry acknowledgement
                if self.speak_callback:
                    self.speak_callback(random.choice(RETRY_PHRASES))

                # Backoff wait
                wait = BASE_BACKOFF * (2 ** (attempt - 2))
                if error_type == "rate_limit":
                    wait += RATE_LIMIT_WAIT
                    logger.info(f"⏳  Rate limited — waiting {wait:.1f}s...")
                elif error_type == "api_error":
                    wait *= 2
                else:
                    logger.info(f"⏳  Waiting {wait:.1f}s before retry...")
                time.sleep(wait)

                # Strategy adjustment
                if error_type == "parse":
                    current_query = self._rephrase(user_input, attempt - 1)
                    logger.info(f"🔁  Rephrased query: \"{current_query}\"")
                else:
                    current_query = user_input  # reset to original

            # Last retry exhausted → use LLM fallback
            if attempt == MAX_RETRIES + 1:
                return self._llm_fallback(user_input)

            try:
                result = self.executor.invoke({"input": current_query})
                output = result.get("output", "").strip()

                if not output:
                    raise ValueError("Empty response from agent")

                logger.info(f"✅  Success on attempt {attempt} | Response: \"{output[:100]}...\"")
                return output

            except Exception as e:
                last_error = e
                logger.error(f"Agent attempt {attempt} failed: {type(e).__name__}: {e}")
                continue

        # Should never reach here, but safety net
        return self._llm_fallback(user_input)

    def clear_memory(self):
        """Reset conversation history."""
        self.memory.clear()
        logger.info("🧹  Conversation memory cleared.")