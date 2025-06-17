from agents import BasicClient, BasicTherapist
from events import TherapySession
from utils import load_json_data, get_model_client
from langfuse.langchain import CallbackHandler

CLIENTS = load_json_data("data/characters/clients.json")
THERAPISTS = load_json_data("data/characters/therapists.json")


if __name__ == "__main__":
    api_type = "LAB"
    # model_name = "gpt-4o"
    model_name = "deepseek/deepseek-r1"
    # api_type = "OR"
    # model_name = "qwen/qwen3-235b-a22b:free"
    # api_type = "OLLAMA"
    # model_name = "qwen3:14b"

    model_client = get_model_client(model_name, api_type)
    session_handler = CallbackHandler()

    client_name = CLIENTS[0]["demographics"]["name"]
    therapist_name = THERAPISTS[0]["demographics"]["name"]

    client = BasicClient(model_client=model_client, data=CLIENTS[0])
    therapist = BasicTherapist(model_client=model_client, data=THERAPISTS[0])
    session = TherapySession(
        client=client,
        therapist=therapist,
        max_turns=12,
        reminder_turn_num=2,
        output_dir=f"data/sessions/{client_name}-{therapist_name}/session_1.json",
    )

    session.graph.invoke(
        input={
            "agenda": None,
            "messages": [],
            "summary": None,
            "homework": None,
            "num_turns": 0,
            "msg": None,
        },
        # config={"callbacks": [session_handler], "recursion_limit": 100},
        config={"recursion_limit": 100},
    )
