import gradio as gr
from main import run_scraper,web_scrapping_logic   

# Define the Gradio interface
iface = gr.Interface(
    fn=run_scraper,
    inputs=gr.Textbox(label="Config path", value="/home/mindmap/Desktop/SLM/unstructured_data/config.yml"),
    outputs=gr.Textbox(label="Scraping Result"),
    title="Web Scraper Prototype",
    description="Prototype your web scraping logic without rewriting it."
)

# Launch the interface
iface.launch()