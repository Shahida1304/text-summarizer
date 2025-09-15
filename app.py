import gradio as gr
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize input text
def summarize_text(text):
    if len(text.strip()) == 0:
        return "⚠️ Please enter some text to summarize."

    # Generate summary
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Build Gradio interface
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=15, 
        placeholder="Paste your long text here..."
    ),
    outputs=gr.Textbox(
        lines=10  
    ),
    title="Text Summarizer App",
    description="Summarize long passages into concise form using Hugging Face's BART model.",
    allow_flagging="never"  
)

if __name__ == "__main__":
    # share=True allows public URL on Hugging Face
    iface.launch()
  

