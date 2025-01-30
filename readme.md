# Real-time Audio Transcription with Google Speech-to-Text and OpenAI Speaker Diarization (POC)

A Python application that performs real-time audio transcription using Google Cloud Platform's Speech-to-Text API via WebSocket protocol, followed by OpenAI-powered speaker diarization for call conversations. The application captures microphone input and provides low-latency transcription with a 30-second recording limit.

## Features

- Real-time audio capture from microphone
- Low-latency transcription using GCP Speech-to-Text
- Speaker diarization using OpenAI for call conversations
- Automatic transcription output saving with timestamps
- 30-second recording limit per session
- Support for multiple languages (configured for English by default)
- Ideal for call center conversations and multi-speaker scenarios

## Prerequisites

- Python 3.7+
- Google Cloud Platform account with Speech-to-Text API enabled
- OpenAI API key
- GCP service account credentials
- PyAudio
- Google Cloud Speech Python client library
- OpenAI Python library

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install google-cloud-speech pyaudio grpcio openai
```
3. Set up your GCP credentials file path in the code
```
export GOOGLE_APPLICATION_CREDENTIALS="path_to_your_key.json"
```
4. Configure your OpenAI API key

## Usage

1. Ensure your GCP credentials and OpenAI API key are properly configured
2. Run the script:
```bash
python google_stt.py
```
3. Start your conversation or call
4. Transcriptions with speaker identification will be saved in the specified output folder

## Usage Notes

- The application automatically stops recording after 30 seconds
- Each recording session creates a new output file
- Audio is processed in real-time, so ensure a stable internet connection
- The microphone must be properly configured and accessible to your system
- You can modify the language code in the script for non-English transcription
- Speaker diarization works best with clear audio and distinct speakers
- Close the application properly to ensure all resources are released

## Output Details

### File Format
- Output files are saved as `.txt` files
- File naming convention: `YYYY-MM-DD_HH-MM-SS.txt`
- Files are stored in the specified output folder

### File Contents
- Timestamp of recording start
- Transcribed text with speaker identification
- Real-time transcription updates
- Automatic notation if 30-second limit is reached
- Each transcription segment is separated by newlines

### Sample Output
```
Recording started at: 2024-01-30_14-30-45
Speaker 1: Hello, how can I help you today?
Speaker 2: I'm calling about my recent order
Speaker 1: Could you please provide your order number?
exceeded 30 sec at: 2024-01-30_14-31-15
```

## Speaker Diarization Process

1. Initial transcription is obtained from Google Cloud Speech-to-Text API
2. Transcribed text is sent to OpenAI for speaker identification
3. OpenAI analyzes conversation patterns and identifies different speakers
4. Final output combines transcription with speaker labels
5. Ideal for:
   - Call center conversations
   - Meeting recordings
   - Multi-speaker interviews
   - Customer service interactions

## Configuration

Key parameters that can be modified:
- `RATE`: Audio sampling rate (default: 16000 Hz)
- `CHUNK`: Audio chunk size
- `language_code`: Transcription language (default: "en-US")
- `output_folder`: Directory for saving transcriptions
- Speaker diarization settings in OpenAI configuration
