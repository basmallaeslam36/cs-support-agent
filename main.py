"""
Customer Support AI Agent — Amazon Bedrock AgentCore + Strands SDK
==================================================================
Deploy:  agentcore configure --entrypoint main.py --name <agent-name>
         agentcore deploy
Invoke:  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Local test: comment out app.run() at the bottom and uncomment main(), then
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── Configuration ────────────────────────────────────────────────────
def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name}. See .env.example for the full list."
        )
    return value


GATEWAY_URL = _require_env("GATEWAY_URL")   # AgentCore Gateway MCP endpoint (ends with /mcp)
KB_ID       = _require_env("KB_ID")         # Bedrock Knowledge Base ID
MEMORY_ID   = _require_env("MEMORY_ID")     # AgentCore Memory ID
REGION      = os.environ.get("AWS_REGION", "us-east-1")


# ── Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── Small helpers for Strands message dicts ──────────────────────────────────
# Strands messages look like {"role": "user", "content": [{"text": "..."}]}.
# Tool results are also "user" messages, but their blocks contain "toolResult".

def _first_text(message):
    """Return the first text block of a message, or None."""
    for block in message.get("content", []):
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    return None


def _is_plain_user_message(message):
    """True for a user message that is not a tool result."""
    if message.get("role") != "user":
        return False
    return not any(
        isinstance(block, dict) and "toolResult" in block
        for block in message.get("content", [])
    )


# ── Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type to namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}
    for strategy in strategies:
        # Newer responses use "namespaceTemplates"; older ones use "namespaces".
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []
        strategy_type = strategy.get("type") or strategy.get("memoryStrategyType")
        if strategy_type and templates:
            namespaces[strategy_type] = templates[0]
    return namespaces


# ── Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)
        self._last_query = None  # the customer's original text, before context is added

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        try:
            messages = event.agent.messages
            if not messages or not _is_plain_user_message(messages[-1]):
                return

            user_query = _first_text(messages[-1])
            if not user_query:
                return
            self._last_query = user_query

            found = []
            for strategy_type, template in self.namespaces.items():
                namespace = template.format(actorId=self.actor_id)
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
                for memory in memories:
                    content = memory.get("content", {}) if isinstance(memory, dict) else {}
                    text = content.get("text", "").strip() if isinstance(content, dict) else ""
                    if text:
                        found.append(f"[{strategy_type}] {text}")

            if found:
                for block in messages[-1]["content"]:
                    if isinstance(block, dict) and "text" in block:
                        block["text"] = (
                            "Customer Context:\n" + "\n".join(found) + "\n\n" + block["text"]
                        )
                        break
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        try:
            messages = event.agent.messages
            customer_query = None
            agent_response = None
            for message in reversed(messages):
                if message.get("role") == "assistant" and not agent_response:
                    agent_response = _first_text(message)
                elif _is_plain_user_message(message) and not customer_query:
                    customer_query = _first_text(message)
                    break

            # Prefer the original text so injected "Customer Context" isn't saved back.
            customer_query = self._last_query or customer_query

            if customer_query and agent_response:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
                )
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        return f"Knowledge base search failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [
        r.get("content", {}).get("text", "")
        for r in results
        if r.get("content", {}).get("text")
    ]
    if not chunks:
        return "No relevant information found in the knowledge base."
    return "\n---\n".join(chunks)


# ── Loyalty Discount Tool (Code Interpreter) ────────────────────────
# ASSUMPTION: 100 points = $1 (1 point = $0.01). Check product_catalog.txt and
# change POINTS_PER_DOLLAR below if the catalog says something different.

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    # Inputs are JSON-encoded so they cannot inject code into the sandbox.
    code = f"""
import json, math

POINTS_PER_DOLLAR = 100
loyalty_points = {int(loyalty_points)}
tier = {json.dumps(str(tier).strip().title())}
order_total = {float(order_total)}
category = {json.dumps(str(product_category).strip().lower())}

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

# Points: cap at 50% of the order, then floor to the nearest 500
max_points = int(order_total * 0.5 * POINTS_PER_DOLLAR)
points_redeemed = (min(loyalty_points, max_points) // 500) * 500
points_discount = round(points_redeemed / POINTS_PER_DOLLAR, 2)

# Tier discount applies to the subtotal after points
subtotal = order_total - points_discount
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal * tier_rate, 2)
final_total = round(subtotal - tier_discount, 2)

result = {{
    "points_redeemed": points_redeemed,
    "points_discount": points_discount,
    "tier": tier,
    "tier_discount_pct": round(tier_rate * 100),
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": round(points_discount + tier_discount, 2),
    "points_earned": math.floor(final_total * earn_rates.get(category, 1)),
    "remaining_points": loyalty_points - points_redeemed,
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {"code": code, "language": "python", "clearContext": True},
            )
            for event in response["stream"]:
                return json.dumps(event["result"])
        return json.dumps({"error": "No result returned from the code interpreter."})

    except Exception as e:
        logger.warning(f"Code Interpreter unavailable, using fallback: {e}")
        rate = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}.get(
            str(tier).strip().title(), 0.0
        )
        discount = round(float(order_total) * rate, 2)
        return json.dumps({
            "points_redeemed": 0,
            "tier_discount_pct": round(rate * 100),
            "tier_discount": discount,
            "final_total": round(float(order_total) - discount, 2),
            "total_savings": discount,
            "remaining_points": int(loyalty_points),
            "note": "Fallback: tier discount only (code interpreter unavailable).",
        })


# ── Agent Entrypoint ─────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a friendly, accurate customer support assistant for an e-commerce platform. "
    "Use the order and refund tools for order tracking, customer lookups, refunds and return "
    "labels. Use search_knowledge_base for product, policy, warranty and loyalty-tier questions. "
    "Use calculate_loyalty_discount for any loyalty discount arithmetic. Use the browser only "
    "when the customer asks for live web information. Use any 'Customer Context' at the start "
    "of a message to personalise your reply. Never invent order, refund or policy details."
)


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = (payload or {}).get("prompt", "").strip()
        if not user_input:
            return "Please include a 'prompt' with your question."
        actor_id = payload.get("customer_id") or "anonymous"
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(actor_id, session_id, memory_client, MEMORY_ID)
        agent_core_browser = AgentCoreBrowser(region=REGION)
        agent_core_browser._nest_asyncio_applied = True  # skip the browser tool's own nest_asyncio patch, which breaks asyncio.timeout on Python 3.14

        tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )
            result = agent(user_input)

        try:
            return result.message["content"][0]["text"]
        except (AttributeError, KeyError, IndexError, TypeError):
            return str(result)

    except Exception as e:
        logger.exception("Agent invocation failed")
        return f"Sorry, I couldn't complete that request. ({e})"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()