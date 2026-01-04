import gradio as gr
import requests

def ask_ncert(question, grade, subject, language):
    data = {"question": question, "grade": grade, "subject": subject, "language": language}
    r = requests.post("http://localhost:8000/ask", json=data)
    return r.json()["answer"]

iface = gr.Interface(
    fn=ask_ncert,
    inputs=["text", "slider", "dropdown", "dropdown"],
    outputs="text"
)
iface.launch()
