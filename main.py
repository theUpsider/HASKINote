import os
import torch
import gradio as gr
from pydub import AudioSegment
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
load_dotenv()

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    raise ValueError("MPS is not available. Please update to CUDA 11.1+")
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model, LORA_WEIGHTS
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )

if device != "cpu":
    model.half()
model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

model_options= ["tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en", "large-v1", "large-v2"]
language_options = ["de", "en", "es", "fr", "it", "nl", "pl"]

# def convert_video_to_audio(video_path, audio_path):
#     video = VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_path)

def convert_audio_to_wav(audio_path, wav_path):
    audio = AudioSegment.from_file(audio_path, format="mp4")
    audio = audio.set_channels(1)  # Ensure mono audio
    audio = audio.set_frame_rate(16000)  # Set frame rate
    audio.export(wav_path, format="wav")

def transcribe_audio(wav_path, model_size, language, progress):
    progress(0.2, desc="Loading model")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    progress(0.3, desc="Transcribing audio")
    segments, info = model.transcribe(wav_path, beam_size=5, language=language)
    if language is None:
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    # append all segment.text together with new line
    text = ""
    progress_status = 0.3
    progress_increment = 0.3 / info.duration
    for segment in segments:
        text += segment.text + "\n"
        start = segment.start
        end = segment.end
        progress_status += progress_increment * (end - start)
        progress(progress_status, desc="Transcribing audio")
    return text

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


def process(video_path, do_summarize, model_size, language, instruction, temperature, progress=gr.Progress()):
    # convert_video_to_audio(video_path, audio_path)
    progress(0, desc="Converting video to audio")
    audio_path = "output.mp4"
    convert_audio_to_wav(video_path.name, audio_path)
    progress(0.1, desc="Transcribing audio")
    text = transcribe_audio(audio_path, model_size, language, progress)
    if do_summarize:
        progress(0.7, desc="Summarizing text")
        summarized = evaluate(instruction, text, temperature=temperature, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=1200)
    else: 
        summarized = ""
    progress(1, desc="Done")
    return text, summarized

iface = gr.Interface(
    fn=process,
    inputs=[gr.inputs.File(label="Video File"), gr.inputs.Checkbox(label="Use instruction LLM (Takes forever!)"), gr.inputs.Dropdown(model_options), gr.inputs.Dropdown(language_options), gr.inputs.Textbox(default="Summarize the following German transcript of a meeting by summarizing it, then structuring it into headings and bullet points to make a meaningful protocol."), gr.inputs.Slider(0, 1, 0.1, label="Temperature", default=0.1)],
    outputs=[gr.outputs.Textbox(label="Transcribed Text"), gr.outputs.Textbox(label="Summarized Text")],
    title="Video to Text Conversion",
    description="Convert a video to text using the Llama model, whisper and some additional libs. Choose a model size, language and instruction. The model will then transcribe the video, summarize the text and structure it into headings and bullet points."
)

if __name__ == "__main__":
    iface.queue(concurrency_count=int(os.getenv("GRADIO_CONCURRENCY_COUNT", 1))).launch(server_name=os.getenv("GRADIO_SERVER_NAME", "localhost"), server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)))
