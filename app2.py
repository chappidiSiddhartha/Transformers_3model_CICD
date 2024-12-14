
import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# Set the title of the app
st.title('Hugging Face Transformers with Streamlit')

# Create a sidebar for selecting model type
model_type = st.sidebar.selectbox(
    "Select a Task", 
    ("Sentiment Analysis", "Text Generation", "Named Entity Recognition")
)

# Load the appropriate model based on the selected task
if model_type == "Sentiment Analysis":
    model = pipeline("sentiment-analysis")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Specify the model name for sentiment analysis
elif model_type == "Text Generation":
    model = pipeline("text-generation")
    model_name = "gpt2"  # Specify the model name for text generation
elif model_type == "Named Entity Recognition":
    model = pipeline("ner")
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Specify the model name for NER

# Save the model to a directory
model_dir = 'outputs/models/'
os.makedirs(model_dir, exist_ok=True)

# Save the model and tokenizer
model.model.save_pretrained(model_dir)  # Save the model weights
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer using the model name
tokenizer.save_pretrained(model_dir)  # Save the tokenizer

# Display the task selected
st.write(f"### Selected Task: {model_type}")

# Create a text input area for the user
user_input = st.text_area('Enter your text here:', height=150)

# Add a button to trigger the output
button = st.button("Analyze Text")

# Handle the button click for inference
if button:
    if user_input:
        with st.spinner("Processing... Please wait."):
            if model_type == "Sentiment Analysis":
                result = model(user_input)
                st.write(f"### Sentiment: {result[0]['label']}")
                st.write(f"**Confidence Score**: {result[0]['score']:.4f}")
            
            elif model_type == "Text Generation":
                result = model(user_input, max_length=50, num_return_sequences=1)
                st.write(f"### Generated Text:")
                st.write(result[0]['generated_text'])
            
            elif model_type == "Named Entity Recognition":
                result = model(user_input)
                st.write("### Named Entities Found:")
                for entity in result:
                    st.write(f"- **Entity**: {entity['word']} | **Label**: {entity['entity']} | **Score**: {entity['score']:.4f}")

    else:
        st.warning("Please enter some text for analysis.")

# Add some footer text or instructions
st.markdown(
    """
    ---
    This app is powered by [Hugging Face Transformers](https://huggingface.co/transformers/) and [Streamlit](https://streamlit.io/).
    """
)

