import os
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

load_dotenv(".env")

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=4096,
    streaming=True,
    base_url=os.environ.get("LAB_BASE_URL"),
    api_key=os.environ.get("LAB_API_KEY"),
)


class SessionState(TypedDict):
    client_data: Dict[str, Any]
    therapist_data: Dict[str, Any]
    agenda: Dict[str, Any]
    messages: List[Dict[str, Any]]
    summary: Optional[str]
    homework: Optional[List[str]]
    num_turns: int
    current_turn: str
    msg: str


def create_agenda(state: SessionState):
    print("Creating agenda")
    agenda = {
        "topics": ["Mental Health", "Stress Management"],
        "goals": ["Improve coping strategies", "Enhance communication skills"],
    }
    return {
        "agenda": agenda,
    }


def generate_therapist_response(state: SessionState):
    print("Generating therapist response")
    msg = {"role": "Therapist", "content": "I am the therapist"}
    print("messages:", state["messages"])
    return {
        "msg": msg,
        "num_turns": state["num_turns"] + 1,
        "messages": state["messages"] + [msg],
    }


def generate_client_response(state: SessionState):
    print("Generating client response")
    msg = {"role": "Client", "content": "I am the client"}
    return {
        "msg": msg,
        "num_turns": state["num_turns"] + 1,
        "messages": state["messages"] + [msg],
    }


def summarize_session(state: SessionState):
    print("Summarizing session")
    return {
        "summary": "This session focused on stress management and coping strategies.",
        "homework": ["Practice mindfulness", "Journal daily"],
    }


def route_turns(state: SessionState):
    num_turns = state.get("num_turns", 0)
    if num_turns > 4:
        print("ending at turn", num_turns)
        return "end"
    return "continue"


simulation_graph = StateGraph(SessionState)
simulation_graph.add_node("create_agenda", create_agenda)
simulation_graph.add_node("therapist_response", generate_therapist_response)
simulation_graph.add_node("client_response", generate_client_response)
simulation_graph.add_node("summarize_session", summarize_session)

simulation_graph.add_edge(START, "create_agenda")
simulation_graph.add_edge("create_agenda", "therapist_response")
simulation_graph.add_edge("therapist_response", "client_response")
# simulation_graph.add_edge("client_response", "check_turn")
simulation_graph.add_conditional_edges(
    "client_response",
    route_turns,
    {
        "end": "summarize_session",
        "continue": "therapist_response",
    },
)
simulation_graph.add_edge("summarize_session", END)

compiled_graph = simulation_graph.compile()

langfuse_handler = CallbackHandler()
results = compiled_graph.invoke(
    input={
        "client_data": None,
        "therapist_data": None,
        "agenda": None,
        "messages": [],
        "summary": None,
        "homework": None,
        "num_turns": 0,
        "current_turn": "",
        "msg": "",
    },
    # config={"callbacks": [langfuse_handler]},
)
