import json
import streamlit as st
from components.cards import SeekerCard

characters = json.loads(open("../data/sources/Patient Psi CM Dataset.json").read())

st.write("# 模拟来访者系统")
st.write(characters[2])
# character_cols = st.columns(3)
# num_characters = 3
# for character in characters[:3]:
#     character_card = SeekerCard(character["name"], character["history"])
#     with character_cols[characters.index(character) % 3]:
#         character_card.display()
