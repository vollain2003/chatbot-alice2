import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the DialoGPT model and tokenizer
model_name = 'microsoft/DialoGPT-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize chat history
chat_history_ids = None

# Streamlit interface
st.title("DialoGPT Chatbot")
st.write("Chat with a model powered by DialoGPT!")

# Initialize session state for chat history
if "chat_history_ids" not in st.session_state:
    st.session_state["chat_history_ids"] = None

user_input = st.text_input("You:", key="input_text")

# Chat button
if st.button("Send"):
    if user_input:
        # Encode the user input and add the end of sentence token
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input to the chat history
        if st.session_state["chat_history_ids"] is not None:
            input_ids = torch.cat([st.session_state["chat_history_ids"], input_ids], dim=-1)

        # Generate a response
        chat_history_ids = model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7
        )

        # Update session state with new chat history
        st.session_state["chat_history_ids"] = chat_history_ids

        # Decode the response and display it
        response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        st.write("Bot:", response)
