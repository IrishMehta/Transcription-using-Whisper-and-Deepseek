
import streamlit as st
from audiorecorder import audiorecorder
import io
import whisper
import requests
from groq import Groq
import librosa
import numpy as np
import contextlib
import webrtcvad
import wave
import torch
from whisper import load_model, transcribe, load_audio
from whisper.audio import log_mel_spectrogram
import os
import collections
import tempfile


def read_wave_resampled(file_like_obj, target_rate=16000):
    file_like_obj.seek(0)
    # Write the file-like object's contents to a temporary WAV file.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_like_obj.read())
        tmp_path = tmp.name

    # Load audio with librosa; this returns a NumPy array and resamples to target_rate.
    audio_np, _ = librosa.load(tmp_path, sr=target_rate, mono=True)
    # Convert the float audio (range [-1, 1]) to int16 PCM format.
    audio_int16 = (audio_np * 32767).astype(np.int16)
    # Convert to bytes.
    pcm_data = audio_int16.tobytes()
    return pcm_data, target_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * frame_duration_ms / 1000.0 * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join(voiced_frames)

def combine_segments(previous_text, current_segment, marker="<SEG>"):
    # Remove any extraneous markers and whitespace
    previous_clean = previous_text.strip() if previous_text else ""
    current_clean = current_segment.strip()

    if previous_clean:
        combined = previous_clean + " " + current_clean
    else:
        combined = current_clean
    return f"{marker} {combined}"

def correct_segments_with_groq(text, api_key):
    # Initialize Groq client (update the initialization per your Groq SDK usage)
    client = Groq(api_key=api_key)

    # Build the prompt; note that we explicitly ask for the output in a strict format.
    prompt = ('''
  Below are examples of transcript segments and their desired outputs. Each example shows how to:

      Correct grammatical errors and restore punctuation.
      Determine if the sentence is complete.
      Format the final response as “Corrected text,True” or “Corrected text,False,” with no additional explanations.

  Example 1:
  Input: "<SEG> mean bird. Ethical considerations and privacy concerns must be addressed to ensure responsible use."
  Output: "Meanwhile, ethical considerations and privacy concerns must be addressed to ensure responsible use.,True"
  Explanation: "mean bird." was changed to "Meanwhile," and the sentence is considered complete.

  Example 2 (incomplete segment):
  Input: "<SEG> As an AI become"
  Output: "As AI systems become,False"
  Explanation: The sentence is incomplete (it stops abruptly), so it is flagged as False.

  Example 3 (complete despite missing terminal punctuation):
  Input: "<SEG> Artificial intelligence and machine learning have revolutionized numerous industries, transforming data processing and predictive analytics"
  Output: "Artificial intelligence and machine learning have revolutionized numerous industries, transforming data processing and predictive analytics.,True"
  Explanation: Even though punctuation was missing, the statement was otherwise complete.

  Example 4 (split sentence across two segments):

      Part 1:
      Input: "<SEG> This involves making the decision making process of AI systems"
      Output: "This involves making the decision-making process of AI systems,False"
      Part 2:
      Input: "<SEG> understandable to users."
      Output after merging: "This involves making the decision-making processes of AI systems understandable to users.,True"
      Explanation: Part 1 on its own was incomplete (False). After merging Part 2, the sentence becomes complete (True).

  Task Instructions:

      Correct any grammatical errors or awkward phrasing.
      Insert necessary punctuation if the sentence is complete.
      Decide if the segment is a complete sentence:
          If complete, end with a period (if necessary) and mark it “True.”
          If incomplete, mark it “False.”
      Output Format:
          Output a single line of text in the form: Corrected text,True or Corrected text,False
          No extra text or explanation should appear.

  Now, please process the following transcript segment. Correct any errors, add punctuation if needed, and decide if it is complete. If it is incomplete, end with “,False.” If it is complete, end with “,True.” Only output this final line with no additional commentary. ''' + str(text) 
    )


    # Make the API call (using the Groq client per their documentation)
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.2,
        max_completion_tokens=1024,
        top_p=0.95,
        stream=False,
        reasoning_format="hidden"
    )

    response_message = completion.choices[0].message.content.strip()
    try:
        # We expect the response to have a comma separator before the completeness flag.
        # Using rsplit ensures that if the corrected text itself contains commas, we still split on the last comma.
        corrected_text, complete_flag_str = response_message.rsplit(",", 1)
        is_complete = complete_flag_str.strip().lower() == "true"
        return corrected_text.strip(), is_complete
    except Exception as e:
        print("Error parsing API response:", e)
        # In case of error, treat the text as incomplete so that it gets buffered.
        return text, False





# Dummy transcription function.
# Replace this with your actual transcription pipeline call.
def transcribe_audio(audio_bytes):
    # In a real scenario, you might save the bytes to a BytesIO and pass it to your model.
    # For example:
    audio_file = io.BytesIO(audio_bytes)
    audio, sample_rate = read_wave_resampled(audio_file, target_rate=16000)
    print(f'The sample rate of the resampled audio file is: {sample_rate}')

    # Initialize VAD
    vad = webrtcvad.Vad(3)  # Set aggressiveness level (0-3)

    # Split audio into segments
    frames = frame_generator(30, audio, sample_rate)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    complete_transcription = []
    buffer_text = ""  # Buffer for incomplete segments
    groq_api_key= "Enter Your API Key"
    model = load_model("turbo")
    for i, segment in enumerate(segments):
        segment_path = f"segment_{i}.wav"
        write_wave(segment_path, segment, sample_rate)

        # Load and transcribe the segment using Whisper
        segment_audio = load_audio(segment_path)
        # Optionally compute mel spectrogram if needed (here not used in the transcription call)
        # mel = log_mel_spectrogram(segment_audio)

        result = transcribe(model, segment_audio)
        segment_text = result["text"].strip()
        os.remove(segment_path)

        # Combine the current segment with any previously buffered incomplete text
        combined_text = combine_segments(buffer_text, segment_text, marker="<SEG>")
        print(f"\nCombined Segment {i+1}:")
        print(combined_text)

        # Send the combined text to Groq for correction and completeness check
        corrected_text, is_complete = correct_segments_with_groq(combined_text, groq_api_key)

        if is_complete:
            # If the sentence is complete, append the corrected text to the final transcription
            complete_transcription.append(corrected_text)
            buffer_text = ""  # Clear the buffer
            print(f"Segment {i+1} is complete. Corrected text:")
            print(corrected_text)
        else:
            # If incomplete, keep the combined text in the buffer for the next iteration
            buffer_text = combined_text
            print(f"Segment {i+1} is incomplete. Buffering for next segment.")

    print("\nFinal Complete Transcription:")
    return complete_transcription

st.title("Voice Recorder Transcription App")
st.markdown("Record your voice and get an instant transcript.")

# Display the recorder component.
# Customize button texts, styles, etc. as needed.
audio = audiorecorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="🛑 Stop Recording",
    custom_style={'color': 'black'},
    start_style={'font-size': '20px'},
)

if audio is not None:
    # Play back the recorded audio.
    st.audio(audio.export().read(), format="audio/wav")

    st.write("Transcribing the audio...")
    # Pass the audio bytes to your transcription pipeline.
    transcript = transcribe_audio(audio.export().read())
    st.markdown("### Transcript")
    st.write(transcript)
