import streamlit as st

def headerhide():
    hide="""
    <style>
    #mainmenu {visiblity:hidden}
    footer{visibility:hidden;}
    header{visibility:hidden;}
    </style>
    """
    st.markdown(hide,unsafe_allow_html=True)
