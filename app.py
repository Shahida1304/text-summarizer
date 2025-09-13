import gradio as gr
from transformers import pipeline

# Load summarization pipeline
# (First run will download the pretrained model)
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
    inputs=gr.Textbox(lines=8, placeholder="Paste your long text here..."),
    outputs="text",
    title="Text Summarizer App",
    description="Summarize long passages into concise form using Hugging Face's BART model.",
)

if __name__ == "__main__":
    # launch() with share=True gives public URL (useful for testing)
    iface.launch()
