"""Test the tutorial card rendering"""
import streamlit as st
from components.tutorial_card import show_tutorial_card
from components.static_demo_page import render_static_demo_page

st.set_page_config(page_title="Tutorial Test", layout="wide")

# Simulate demo mode
if 'demo_active' not in st.session_state:
    st.session_state.demo_active = False
if 'demo_step' not in st.session_state:
    st.session_state.demo_step = 0

if st.button("Start Tutorial"):
    st.session_state.demo_active = True
    st.session_state.demo_step = 0
    st.rerun()

if st.session_state.demo_active:
    st.write("Demo is active!")
    st.write(f"Step: {st.session_state.demo_step}")

    # Render demo page
    render_static_demo_page(st.session_state.demo_step)

    # Show tutorial card
    show_tutorial_card(st.session_state.demo_step)

    st.stop()

st.write("Not in demo mode - click the button above")
