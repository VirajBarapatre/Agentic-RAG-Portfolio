"""The agent: an LLM that decides which tools to call, in a LangGraph loop."""

from groq import BadRequestError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import REFUSAL_PHRASE
from src.llm import get_llm
from src.tools.agent_tools import ALL_TOOLS

MAX_TOOL_CALLS = 4  # per question

SYSTEM_PROMPT = f"""You answer questions about Viraj Barapatre's projects using tools.

Tools:
- rag_search: details from project READMEs (what a project does, tech stack, dataset, results).
- sql_query: the `projects` table, for lists, counts, filters and comparisons across projects.
- calculator: any arithmetic.

Rules:
1. Answer only from tool results in this conversation. Never use outside knowledge about Viraj's projects.
2. If the tool results do not contain the answer, reply exactly "{REFUSAL_PHRASE}" Do not guess.
3. If the question is not about Viraj's projects at all, reply exactly "{REFUSAL_PHRASE}" without calling any tool.
4. A number in a search passage belongs to the project in that passage's source file. Never attribute it to a different project.
5. Use the calculator for every calculation.
6. Keep answers short and say which project each fact comes from.
"""

LIMIT_NOTE = (
    "Tool-call limit reached. Answer now using only the tool results above, "
    f'or reply exactly "{REFUSAL_PHRASE}"'
)


def _tool_calls_this_turn(messages: list) -> int:
    """Count tool calls made since the user's latest question."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            count += len(message.tool_calls)
    return count


def build_agent():
    """Build the agent graph: agent node <-> tools node, with memory."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    llm_no_tools = llm.bind_tools(ALL_TOOLS, tool_choice="none")

    def final_answer(messages: list) -> AIMessage:
        """At the tool limit: ask for a final answer; if the model still wants tools, refuse."""
        try:
            response = llm_no_tools.invoke(messages + [SystemMessage(LIMIT_NOTE)])
        except BadRequestError:
            # Groq rejects the request when the model calls a tool despite tool_choice="none".
            return AIMessage(REFUSAL_PHRASE)
        if response.tool_calls:
            return AIMessage(REFUSAL_PHRASE)
        return response

    def agent_node(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        if _tool_calls_this_turn(state["messages"]) >= MAX_TOOL_CALLS:
            response = final_answer(messages)
        else:
            response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_after_agent(state: MessagesState) -> str:
        """If the LLM asked for tools, run them; otherwise the answer is final."""
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=InMemorySaver())


def run_question(agent, question: str, thread_id: str) -> dict:
    """Ask one question and return the answer plus a record of this turn."""
    result = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]
    last_question = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn = messages[last_question + 1:]

    ai_messages = [m for m in turn if isinstance(m, AIMessage)]
    tool_calls = [call for m in ai_messages for call in m.tool_calls]
    usage = [m.usage_metadata for m in ai_messages if m.usage_metadata]
    return {
        "answer": turn[-1].content,
        "tools_used": [call["name"] for call in tool_calls],
        "tool_args": [call["args"] for call in tool_calls],
        "tool_outputs": [m.content for m in turn if isinstance(m, ToolMessage)],
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
    }


def ask(agent, question: str, thread_id: str) -> str:
    """Send one question in a conversation (thread) and return only the answer text."""
    return run_question(agent, question, thread_id)["answer"]