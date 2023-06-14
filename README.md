# Speech to Text Conversion Application

This application converts spoken words in an mp4 video file to text. It first converts the video to a 16kbits wav audio file, and then uses the Whisper ASR system to transcribe the speech to text. Then it uses Llama to run a language model on the text to correct any errors in the transcription.

## Dependencies

It also requires `ffmpeg` to be installed and available in your PATH.

You can install the Python dependencies with conda:

```sh
conda env create -f environment.yml
```

For `ffmpeg`, you can install it on Ubuntu using apt:

```sh
sudo apt update
sudo apt install ffmpeg
```

On MacOS, you can use Homebrew:

```sh
brew install ffmpeg
```

On Windows, you can download the binaries from the ffmpeg website and add it to your PATH.

## How to Run

To run this application, you can use the following command:

```sh
python main.py
```

This will open a gradio interface where you can upload a video file and see the transcription.

## License

This project is licensed under the terms of the MIT license.
