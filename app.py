import gradio as gr
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if len(text.strip()) == 0:
        return "⚠️ Please enter some text to summarize."
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Gradio interface
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=15, placeholder="Paste your long text here..."),
    outputs=gr.Textbox(lines=10, placeholder="Summary will appear here..."),
    title="Text Summarizer App",
    description="Summarize long passages using Hugging Face BART model.",
    allow_flagging="never"
)

# For HF Spaces, remove share=True
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
