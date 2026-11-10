"""Interactive chat with the agent. Shows every tool call and result.

Run from the project root:
    python -m scripts.chat
Commands: 'new' starts a fresh conversation, 'quit' exits.
"""

import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import build_agent

PREVIEW_CHARS = 300  # how much of each tool result to show


def print_turn(messages: list) -> None:
    """Print the tool calls and results since the latest question, then the answer."""
    last_question = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    for message in messages[last_question + 1:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  -> {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            preview = message.content[:PREVIEW_CHARS].replace("\n", " ")
            print(f"  <- {preview}")
    print(f"\nAgent: {messages[-1].content}\n")


def main() -> None:
    agent = build_agent()
    thread_id = str(uuid.uuid4())
    print("Ask about Viraj's projects. 'new' = fresh conversation, 'quit' = exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "new":
            thread_id = str(uuid.uuid4())
            print("(New conversation.)\n")
            continue
        if not question:
            continue
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(question)]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as err:
            print(f"\n[Error: {type(err).__name__}: {err}]\n")
            continue
        print_turn(result["messages"])


if __name__ == "__main__":
    main()