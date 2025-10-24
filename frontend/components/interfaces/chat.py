import time
import asyncio
import streamlit as st
from typing import Dict, List
from apis.chat import send_message_to_chat_api


class ChatInterface:
    def __init__(self, title: str = ""):
        self.title = title
        self.response = ""

    def display(self):
        chatbox = st.container(height=500, border=True)
        chat_container = chatbox.container(height=400, border=False)

        input_container = chatbox.container(height=100, border=False)
        with chatbox:
            if self.title:
                st.title(self.title)

            with chat_container:
                for msg in st.session_state.get("chat_messages", []):
                    for role, content in msg.items():
                        st.chat_message(role).markdown(content)
            with input_container:
                cols = st.columns([0.88, 0.12])
                with cols[0]:
                    if user_msg := st.chat_input("Your message"):
                        with chat_container:
                            st.chat_message("user").markdown(user_msg)
                        self.response = asyncio.run(
                            self.send_message({"user": user_msg})
                        )
                        with chat_container:
                            st.chat_message("assistant").write_stream(
                                self.stream_response
                            )
                with cols[1]:
                    if st.button("Clear"):
                        st.session_state["chat_messages"] = []
                        st.rerun()

    def add_message(self, msg: Dict[str, str]):
        messages = st.session_state.get("chat_messages", [])
        st.session_state["chat_messages"] = messages + [msg]

    async def send_message(self, msg: Dict[str, str]):
        self.add_message(msg)
        print("sending message")
        response = await send_message_to_chat_api(msg.get("user", ""))
        self.add_message({"assistant": response})
        return response

    def stream_response(self):
        for chunk in self.response["content"].split():
            yield chunk + " "
            time.sleep(0.15)
