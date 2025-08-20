import os
import gradio as gr
from brain_of_the_doc import analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doc import text_to_speech

SYSTEM_PROMPT = (
    "You have to act as a professional doctor, i know you are not but this is for learning purpose. "
    "What's in this image? Do you find anything medically wrong? If you make a differential, suggest some remedies. "
    "Do not add any number or special character in your response. Your response should be in one long paragraph. "
    "Always answer as if you are talking to a real person. Do not say 'In the image I see' but say "
    "'with what I see, I think you have...'. Do not respond as an AI model; mimic an actual doctor's tone. "
    "Keep your answer concise (max two sentences), no preamble—start right away."
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
STT_MODEL = "whisper-large-v3"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def process_inputs(audio_path: str, image_path: str):
    # 1) Transcribe the patient's voice message (if provided)
    stt_text = ""
    if audio_path:
        stt_text = transcribe_with_groq(STT_MODEL, audio_path, GROQ_API_KEY)

    # 2) Build query by combining system_prompt + patient's text
    query = SYSTEM_PROMPT
    if stt_text:
        query = f"{SYSTEM_PROMPT}\n\nPatient says: {stt_text.strip()}"

    # 3) Analyze the image with the query
    if not image_path:
        doctor_response = "I couldn't find an image. Please provide a clear photo of the concern."
    else:
        doctor_response = analyze_image_with_query(
            query=query,
            model=VISION_MODEL,
            image_path=image_path,
            api_key=GROQ_API_KEY,
        )

    # 4) Convert doctor's text to speech
    audio_reply_path = text_to_speech(doctor_response, output_filepath="doctor_reply.mp3")
    return stt_text, doctor_response, audio_reply_path

with gr.Blocks(title="DiagnoVisionAI") as iface:
    gr.Markdown("# DiagnoVisionAI")
    gr.Markdown("Upload an image of the concern and optionally speak your question.")

    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak your question (optional)")
        img = gr.Image(type="filepath", label="Upload image")

    with gr.Row():
        stt_out = gr.Textbox(label="Speech to Text")
    with gr.Row():
        doc_out = gr.Textbox(label="Doctor's Response")
    with gr.Row():
        tts_out = gr.Audio(label="Doctor's Voice Reply")

    submit = gr.Button("Analyze")
    submit.click(fn=process_inputs, inputs=[mic, img], outputs=[stt_out, doc_out, tts_out])

if __name__ == "__main__":
    iface.launch(debug=True)
