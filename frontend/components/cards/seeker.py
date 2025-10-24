import streamlit as st


class SeekerCard:
    def __init__(self, name: str, desc: str):
        self.name = name
        self.desc = desc

    def display(self):
        card = st.container(border=True, width=300, height=200)
        with card:
            st.subheader(self.name)
            st.write(self.desc)
        return card
