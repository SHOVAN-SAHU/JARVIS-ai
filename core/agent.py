"""
core/agent.py
JARVIS brain — LangChain ReAct agent on Groq LLaMA 3.3.

Key improvements:
  - Strict tool-use policy in system prompt (no unnecessary searches)
  - Retry system with exponential backoff + error-type strategies
  - LLM-direct fallback when all retries fail
"""
import logging
import time
import random
from typing import Callable

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from config import settings
from tools import get_all_tools

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_BACKOFF = 1.0
RATE_LIMIT_WAIT = 8.0

RETRY_PHRASES = [
    "One moment — running into a snag. Let me try again.",
    "Give me a second, attempting a different approach.",
    "Encountered an issue. Retrying now, sir.",
]
FALLBACK_PHRASES = [
    "My search tools aren't available right now, so I'll answer from what I know.",
    "I'll answer directly since my tools are having trouble.",
]

# ── System Prompt ─────────────────────────────────────────────────────────────
# The single most important change: explicit tool-use rules for the LLM.
JARVIS_SYSTEM_PROMPT = """You are JARVIS, an advanced AI assistant — precise, witty, and efficient.
Your responses are spoken aloud, so keep them natural and concise.
Your master's name is Shovan. You were created by Shovan.

RULES:
- Never mention tool names, APIs, or internal processes to the user.
- Never say "I don't have information but you can search" — just search silently and answer.
- Use tools silently. The user only ever hears your Final Answer.
- Call web search tool at most ONCE per question. After getting results, go straight to Final Answer.
- Use other tools until you get you result or iteration limit is done.
- For anything time-sensitive, recent, or real-time — search first, then answer.
- For everything else — answer directly from your knowledge.

Available tools:
{tools}

Format (follow EXACTLY):
Question: the input question
Thought: do I need a tool?
Action: tool name (one of [{tool_names}])
Action Input: input to the tool
Observation: tool result
Thought: I now know the final answer
Final Answer: natural, conversational spoken response

If no tool needed:
Thought: I can answer directly.
Final Answer: <your answer>

{chat_history}
Question: {input}
Thought: {agent_scratchpad}"""


def _classify_error(e: Exception) -> str:
    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg:
        return "rate_limit"
    if "connection" in msg or "timeout" in msg:
        return "network"
    if "parsing" in msg or "could not parse" in msg or "output" in msg:
        return "parse"
    return "unknown"


class JarvisAgent:
    def __init__(self, speak_callback: Callable[[str], None] | None = None):
        self.speak_callback = speak_callback
        self.tools = get_all_tools()
        self._build_agent()

    def _build_agent(self):
        self._llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_LLM_MODEL,
            temperature=0.5,
            max_tokens=1024,
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=False,
        )

        agent = create_react_agent(
            llm=self._llm,
            tools=self.tools,
            prompt=PromptTemplate.from_template(JARVIS_SYSTEM_PROMPT),
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
            f"🧠  Agent ready | {settings.GROQ_LLM_MODEL} | "
            f"Tools: {[t.name for t in self.tools]}"
        )

    def _maybe_speak_action(self, query: str):
        """Speak a pre-action phrase only for queries that clearly need a tool."""
        if not self.speak_callback:
            return
        q = query.lower()
        hints = {
            ("search", "look up", "find online", "google", "latest news", "current price",
             "exchange rate", "stock", "score", "weather"): "Searching the web for that now.",
            ("what time", "current time", "what's the date", "today's date"): "Checking the time for you.",
            ("cpu", "battery", "ram", "memory usage", "disk space"): "Checking your system now.",
        }
        for keywords, phrase in hints.items():
            if any(k in q for k in keywords):
                self.speak_callback(phrase)
                return

    def _llm_fallback(self, query: str) -> str:
        logger.info("🔄  Direct LLM fallback (no tools)...")
        if self.speak_callback:
            self.speak_callback(random.choice(FALLBACK_PHRASES))
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            msgs = [
                SystemMessage(content="You are JARVIS, a helpful AI assistant. "
                              "Answer concisely and naturally — your response will be spoken aloud."),
                HumanMessage(content=query),
            ]
            response = self._llm.invoke(msgs)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return ("I'm experiencing technical difficulties right now, sir. "
                    "All retry attempts exhausted. Please try again in a moment.")

    def _rephrase(self, query: str, attempt: int) -> str:
        prefixes = ["Please answer this directly: ", "In simple terms, ", "Briefly, "]
        return prefixes[(attempt - 1) % len(prefixes)] + query

    def think(self, user_input: str) -> str:
        logger.info(f"🤔  Thinking: \"{user_input}\"")
        self._maybe_speak_action(user_input)

        current_query = user_input
        last_error = None

        for attempt in range(1, MAX_RETRIES + 2):
            is_retry = attempt > 1

            if is_retry:
                error_type = _classify_error(last_error)
                logger.warning(f"⚠️  Retry {attempt-1}/{MAX_RETRIES} | {error_type}")

                if self.speak_callback:
                    self.speak_callback(random.choice(RETRY_PHRASES))

                wait = BASE_BACKOFF * (2 ** (attempt - 2))
                if error_type == "rate_limit":
                    wait += RATE_LIMIT_WAIT
                elif error_type == "unknown":
                    wait *= 2
                logger.info(f"⏳  Waiting {wait:.1f}s...")
                time.sleep(wait)

                if error_type == "parse":
                    current_query = self._rephrase(user_input, attempt - 1)
                else:
                    current_query = user_input

            if attempt == MAX_RETRIES + 1:
                return self._llm_fallback(user_input)

            try:
                result = self.executor.invoke({"input": current_query})
                output = result.get("output", "").strip()
                if not output:
                    raise ValueError("Empty agent response")
                logger.info(f"✅  Done (attempt {attempt}): \"{output[:100]}...\"")
                return output
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt} failed: {type(e).__name__}: {e}")

        return self._llm_fallback(user_input)

    def clear_memory(self):
        self.memory.clear()
        logger.info("🧹  Memory cleared.")