import gradio as gr

def process_query(query: str) -> str:
    """
    Simple function to log and return the query.
    """
    print(f"Received query: {query}")  # Log to terminal for debugging
    return f"Query received: {query}"  # Respond to Gradio

def gradio_app():
    """
    Launch a Gradio app for testing query submission.
    """
    print("Launching Gradio app...")

    # Define Gradio interface
    interface = gr.Interface(
        fn=process_query,
        inputs=gr.Textbox(lines=2, label="Enter your query"),
        outputs=gr.Textbox(label="Backend Response"),
        title="Test Query Submission",
        description="Enter a query and see the backend response.",
    )

    # Launch the app
    interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    gradio_app()
