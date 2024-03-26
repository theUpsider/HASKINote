import os
import re
import gradio as gr
from pydub import AudioSegment
import whisperx
from dotenv import load_dotenv
from whisper_diarization.transcription_helpers import transcribe
from whisper_diarization.helpers import *

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel


load_dotenv()


model_options = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
]
language_options = ["de", "en", "es", "fr", "it", "nl", "pl"]

# def convert_video_to_audio(video_path, audio_path):
#     video = VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_path)


def convert_file_to_wav(audio_path, wav_path):
    file_format = audio_path.split(".")[-1]
    audio = AudioSegment.from_file(audio_path, format=file_format)
    audio = audio.set_channels(1)  # Ensure mono audio
    audio = audio.set_frame_rate(16000)  # Set frame rate
    audio.export(wav_path, format="wav")


def transcribe_audio(wav_path, model_size, language, progress):
    progress(0.2, desc="Loading model")
    print("Loading model")
    whisper_results, language = transcribe(
        wav_path,
        language,
        model_size,
        "float16",
        False,
        "cuda",
    )
    print(whisper_results)
    progress(0.5, desc="Aligning text")
    if language in wav2vec2_langs:
        alignment_model, metadata = whisperx.load_align_model(
            language_code=language, device="cuda"
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, wav_path, "cuda"
        )
        word_timestamps = filter_missing_timestamps(
            result_aligned["word_segments"],
            initial_timestamp=whisper_results[0].get("start"),
            final_timestamp=whisper_results[-1].get("end"),
        )
        # clear gpu vram
        del alignment_model
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append(
                    {"word": word[2], "start": word[0], "end": word[1]}
                )

    # convert audio to mono for NeMo combatibility
    sound = AudioSegment.from_file(wav_path).set_channels(1)
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")
    progress(0.8, desc="Diarizing text")
    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")
    msdd_model.diarize()

    progress(0.9, desc="Restoring punctuation")
    # Reading timestamps <> Speaker Labels mapping
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {language} language. Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    sentences_speaker_mapping = get_sentences_speaker_mapping(wsm, speaker_ts)
    previous_speaker = sentences_speaker_mapping[0]["speaker"]

    text = ""

    progress(1, desc="Done")
    for sentence_dict in sentences_speaker_mapping:
        speaker = sentence_dict["speaker"]
        sentence = sentence_dict["text"]
        # If this speaker doesn't match the previous one, start a new paragraph
        if speaker != previous_speaker:
            text += "\n\n" + speaker + ": "
            previous_speaker = speaker

        text += sentence + " "

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


def process(
    file_path,
    do_summarize,
    model_size,
    language,
    instruction,
    temperature,
    progress=gr.Progress(),
):
    # convert_video_to_audio(video_path, audio_path)
    progress(0, desc="Converting audio")
    audio_path = "output.mp4"
    convert_file_to_wav(file_path.name, audio_path)
    progress(0.1, desc="Transcribing audio")
    text = transcribe_audio(audio_path, model_size, language, progress)
    if False:
        progress(0.7, desc="Summarizing text")
        summarized = evaluate(
            instruction,
            text,
            temperature=temperature,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=1200,
        )
    else:
        summarized = ""
    progress(1, desc="Done")
    return text, summarized


iface = gr.Interface(
    fn=process,
    inputs=[
        gr.components.File(label="Video/Audio File"),
        gr.components.Checkbox(label="Use instruction LLM (Takes forever!)"),
        gr.components.Dropdown(model_options),
        gr.components.Dropdown(language_options),
        gr.components.Textbox(
            value="Summarize the following German transcript of a meeting by summarizing it, then structuring it into headings and bullet points to make a meaningful protocol."
        ),
        gr.components.Slider(0, 1, 0.7, label="Temperature"),
    ],
    outputs=[
        gr.components.Textbox(label="Transcribed Text"),
        gr.components.Textbox(label="Summarized Text"),
    ],
    title="Video to Text Conversion",
    description="Convert a video to text using the Llama model, whisper and some additional libs. Choose a model size, language and instruction. The model will then transcribe the video, summarize the text and structure it into headings and bullet points.",
)


def handle_auth(username, password):
    return username == os.getenv("GRADIO_USERNAME") and password == os.getenv(
        "GRADIO_PASSWORD"
    )


if __name__ == "__main__":
    iface.queue().launch(
        auth=handle_auth,
        server_name=os.getenv("GRADIO_SERVER_NAME", None),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )
