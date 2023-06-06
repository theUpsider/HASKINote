# Speech to Text Conversion Application

This application converts spoken words in an mp4 video file to text. It first converts the video to a 16kbits wav audio file, and then uses the Whisper ASR system to transcribe the speech to text.

## Dependencies

This application depends on the following Python libraries:

- moviepy
- pydub
- ctranslate2

It also requires `ffmpeg` to be installed and available in your PATH.

You can install the Python dependencies with pip:

```sh
pip install moviepy pydub ctranslate2
```
or use the requirements.txt file
```sh
pip install -r requirements.txt
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

This will take the input video file `"input.mp4"`, convert it to an audio file `"output.mp4"`, convert that audio file to a wav file `"output.wav"`, and finally, use the Whisper ASR system to transcribe the audio into text.

You need to replace `"path_to_your_model"` in `main.py` with the actual path to your Whisper ASR model.

## License

This project is licensed under the terms of the MIT license.

## Contact

For any questions, you can reach us at [your-email@example.com](mailto:your-email@example.com).
```

Remember to replace `[your-email@example.com](mailto:your-email@example.com)` with your actual email address or other contact information. If you don't want to provide an email address, you could instead instruct people to open issues in the GitHub repository if they have questions or problems.

Also, replace the license with the one that suits your needs and preferences.