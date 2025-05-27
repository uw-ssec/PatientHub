import os
import json
import argparse
import streamlit as st
from components.character import Character
from agents import BasicClient, BasicTherapist
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv(".env")

model = ChatOpenAI(
    model=os.environ.get("MODEL_NAME"),
    base_url=os.environ.get("BASE_URL"),
    api_key=os.environ.get("API_KEY"),
    temperature=0.6,
)


CLIENTS = json.load(open("data/characters/clients.json"))
THERAPISTS = json.load(open("data/characters/therapists.json"))


st.set_page_config(layout="wide")
st.html("<style>" + open("./assets/styles.css").read() + "</style>")


state = {"messages": [], "mental_state": {}, "client_mental_state": {}}

for k, v in state.items():
    if k not in st.session_state:
        st.session_state[k] = v


def generate_response(agent, msg, idx=0):
    if not idx:
        res, mental_state = agent.generate_response()
    else:
        res, mental_state = agent.receive_message(msg)
    st.session_state.messages.append({"role": agent.role, "content": res})
    if agent.role == "client":
        st.session_state.mental_state = mental_state
    elif agent.role == "therapist":
        st.session_state.client_mental_state = mental_state
    return res


def reset_simulation(client, therapist):
    client.reset_messages()
    client.set_prompt(CLIENTS[0])
    therapist.reset_messages()
    therapist.set_prompt(THERAPISTS[0])
    st.session_state.messages = []


def start_simulation(patient, therapist, chat_container):
    # reset_simulation(patient, therapist)
    conv_len = 4
    prev_res = ""
    for i in range(conv_len):
        role = "patient" if i % 2 == 0 else "therapist"
        prev_res = generate_response(
            therapist if i % 2 == 0 else patient, prev_res, idx=i
        )
        chat_container.chat_message(role).markdown(prev_res)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="openai")
    parser.add_argument("--model_path", type=str, default="gpt-4o")
    parser.add_argument("--lang", type=str, default="all", choices=["en", "zh"])
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()

    client = BasicClient(model, CLIENTS[0])
    therapist = BasicTherapist(model, THERAPISTS[0])

    # st.write(client.sys_prompt)
    st.write(therapist.sys_prompt)
    client_col, chat_col, therapist_col = st.columns([1, 3, 1])
    client_obj = Character(data=CLIENTS[0])
    client_obj.get_mental_state(
        st.session_state.mental_state, st_column=client_col, title="Client View"
    )

    therapist_obj = Character(data=THERAPISTS[0])
    therapist_obj.get_mental_state(
        st.session_state.client_mental_state, therapist_col, "Therapist View"
    )

    chatbox_container = chat_col.container(height=600, border=True, key="chatbox")
    chat_container = chatbox_container.container(
        height=600, border=False, key="chat_box"
    )

    chat_col.button(
        "Start Simulation",
        on_click=start_simulation,
        args=(client, therapist, chat_container),
    )
    chat_col.button(
        "Reset Simulation",
        on_click=reset_simulation,
        args=(client, therapist),
    )

    with chatbox_container:
        with chat_container:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).markdown(msg["content"])
