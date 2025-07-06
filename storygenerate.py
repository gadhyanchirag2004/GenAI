import torch
import gradio as gr
from transformers import pipeline, set_seed

# Load GPT-2 text generation pipeline
story_gen = pipeline("text-generation", model="gpt2")
set_seed(42)  # Optional: for reproducibility

# Define the generation function
def generate_story(prompt):
    output = story_gen(prompt, max_length=200, do_sample=True, temperature=0.9)
    return output[0]['generated_text']

# Gradio interface
gr.close_all()
demo = gr.Interface(
    fn=generate_story,
    inputs=[gr.Textbox(label="Enter your story prompt", lines=4)],
    outputs=[gr.Textbox(label="AI Generated Story", lines=10)],
    title="📝 AI Story Generator",
    description="Enter a prompt and watch the AI craft a short creative story.",
)

demo.launch()
