import torch
import gradio as gr
from transformers import pipeline, set_seed

# Set seed for reproducibility
set_seed(42)

# Load GPT-2 text generation pipeline with small model for faster response
story_gen = pipeline("text-generation", model="gpt2")

def generate_story(prompt):
    # Generate story using tuned parameters for relevance and coherence
    output = story_gen(
        prompt,
        max_length=150,            # Limit length for concise stories
        do_sample=True,            # Enable sampling for creativity
        temperature=0.7,           # Balanced creativity and focus
        top_k=50,                  # Limit to top 50 token choices each step
        top_p=0.9,                 # Nucleus sampling for diversity control
        repetition_penalty=1.2,    # Reduce repetitive text generation
        num_return_sequences=1,    # Only 1 story output per prompt
        pad_token_id=50256         # GPT-2 pad token ID to avoid warnings
    )
    # Return generated text string
    return output[0]['generated_text']

# Close any old Gradio interfaces if running multiple times interactively
gr.close_all()

# Setup Gradio UI Interface with clear titles and labels
demo = gr.Interface(
    fn=generate_story,
    inputs=[gr.Textbox(label="Enter your story prompt", lines=4, placeholder="Start your story here...")],
    outputs=[gr.Textbox(label="AI Generated Story", lines=10)],
    title="📝 AI Story Generator",
    description="Enter a prompt and watch the AI craft a short, creative, and relevant story."
)

# Launch with public shareable URL for easy demonstration and access
demo.launch(share=True)

