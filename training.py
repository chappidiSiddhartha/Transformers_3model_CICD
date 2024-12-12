import os
import argparse
import joblib
import streamlit as st
from azureml.core import Run, Workspace
from transformers import pipeline
import json

# Initialize Azure ML Run
run = Run.get_context()
workspace = run.experiment.workspace

# Set the title of the app
st.title('Hugging Face Transformers with Streamlit and Azure ML')

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Streamlit App with Azure ML Integration')
    parser.add_argument('--model_path', type=str, help='Path to save serialized model', default='./model_metas')
    parser.add_argument('--artifact_path', type=str, help='Path to save output artifacts', default='./outputs')
    return parser.parse_args()

args = parse_args()
os.makedirs(args.model_path, exist_ok=True)
os.makedirs(args.artifact_path, exist_ok=True)

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

# Serialize the model to save it for Azure ML tracking
model_save_path = os.path.join(args.model_path, f"{model_type.replace(' ', '_')}_model.pkl")
joblib.dump(model, model_save_path)
run.upload_file(name=f"{model_type.replace(' ', '_')}_model.pkl", path_or_stream=model_save_path)
st.write(f"### Model Serialized and Saved to: {model_save_path}")

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
            output_path = os.path.join(args.artifact_path, f"{model_type.replace(' ', '_')}_output.json")
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

            # Save and log results
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            run.upload_file(name=f"{model_type.replace(' ', '_')}_output.json", path_or_stream=output_path)

    else:
        st.warning("Please enter some text for analysis.")

# Log and complete the Azure ML run
run.log("Selected Task", model_type)
run.tag("StreamlitApp")
run.complete()

# Add some footer text or instructions
st.markdown(
    """
    ---
    This app is powered by [Hugging Face Transformers](https://huggingface.co/transformers/) and [Streamlit](https://streamlit.io/), integrated with [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/).
    """
)
