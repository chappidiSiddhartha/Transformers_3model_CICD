
import streamlit as st
from transformers import pipeline

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
elif model_type == "Text Generation":
    model = pipeline("text-generation")
elif model_type == "Named Entity Recognition":
    model = pipeline("ner")

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
