import streamlit as st
from components.interfaces import ChatInterface


st.write("# Patient-$$\\Psi$$")

chat_interface = ChatInterface()
chat_interface.display()
