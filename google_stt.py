import datetime
import sys
import os
import queue
import threading
import grpc
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
import pyaudio

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

credentials = service_account.Credentials.from_service_account_file(
    "C:\AI_ML\google_stt_poc\credentials\da21e352e56a2.json")
client = speech.SpeechClient(credentials=credentials)
# # Load credentials
# credentials = service_account.Credentials.from_service_account_file(
#     os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# )

client = speech.SpeechClient(credentials=credentials)


# Configure microphone stream
class MicrophoneStream:
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk


output_folder = "C:\AI_ML\google_stt_poc\STT_output"


# Define transcription function
def listen_print_loop(responses, file_name,start_time,stream):
    for response in responses:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(duration)
        if duration > 30:
            print("Audio recording exceeded 30 seconds.")
            with open(file_name, 'a') as file:
                file.write(f"\nexceeded 30 sec at:{end_time.strftime('%Y-%m-%d_%H-%M-%S')}")
            stream.closed = True
            break
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        # print(f"Transcript: {transcript}")

        if result.is_final:
            # print("\nFinalized: ", transcript)
            # print("======================================================")

            # Save the content to the .txt file
            with open(file_name, 'a') as file:
                file.write(f"\n{transcript}")


# Main function
def main():
    start_time = datetime.datetime.now()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(output_folder, f"{timestamp}.txt")
    with open(file_name, 'w') as file:
        file.write(f"Recording started at: {timestamp}\n")
    language_code = "en-US"  # Replace with "he-IL" for Hebrew
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_speaker_diarization=False,  # Set to True for speaker differentiation
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(
            config=streaming_config,
            requests=requests,
        )

        listen_print_loop(responses, file_name,start_time,stream)


if __name__ == "__main__":
    main()
