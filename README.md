# Transcription-using-Whisper-and-Deepseek
This repository tries to improve the accuracy of transcription by combining Whisper with WebRTC Voice Activity Detector and DeepSeek. The goal is to figure out a way to use the power of general speech pauses for separating out sentences and an LLM for grammar correction to correct any mistakes that transcription might have.

## Overview

**Goal**: Provide a user-friendly interface where you can record an audio snippet directly from the browser, automatically detect speech segments, transcribe them, and intelligently merge/correct them into a coherent final transcript.

**Core Steps**:
1. **Record** using an in-browser audio recorder.
2. **Split** audio into speech segments using WebRTC VAD (Voice Activity Detection).
3. **Transcribe** segments using a local Whisper model (OpenAI’s speech recognition).
4. **Combine** partial or incomplete segments.
5. **Refine** using Groq’s AI-based grammar and completeness checker.

## Architecture & Workflow

1. **Audio Capture**  
   - [Streamlit](https://streamlit.io/) plus the [`audiorecorder`](https://pypi.org/project/audiorecorder/) library handle in-browser recording.
   - The app receives the raw audio data as bytes.

2. **Audio Resampling**  
   - Using [`librosa.load()`](https://librosa.org/doc/main/generated/librosa.load.html) to load and resample the audio to 16 kHz, making it consistent with Whisper’s typical requirements.

3. **Voice Activity Detection**  
   - [`webrtcvad.Vad`](https://pypi.org/project/webrtcvad/) is used at an aggressiveness level of `3` (most sensitive).  
   - Audio is chunked into frames (`30 ms` each), and VAD determines which frames contain speech.

4. **Segment Collection**  
   - As soon as a segment of speech is detected, frames are concatenated. When silent frames dominate, that segment is finalized and saved as a temporary WAV file.

5. **Whisper Transcription**  
   - For each segment, the script calls `whisper.transcribe()` on the local model. The recognized text is appended to a running transcript buffer.

6. **Combining & Correcting**  
   - **Combining**: If Whisper’s segment transcription appears incomplete, it’s buffered until the next segment.  
   - **Groq**: The incomplete/complete segments are sent to Groq’s large language model, which:  
     1. Corrects grammar.  
     2. Adds punctuation.  
     3. Decides if the segment is complete.  
   - **Result**: If Groq marks a segment as complete, it’s appended to the final transcript. If it’s incomplete, it’s merged with the next segment until a full statement is formed.

7. **Final Output**  
   - Once all segments are processed, you get a coherent, fully punctuated transcript.

## Dependencies & Installation**
`pip install streamlit audiorecorder whisper requests groq librosa numpy webrtcvad torch`


Recording and Transcription

    Press “🎤 Start Recording” and speak.
    Press “🛑 Stop Recording” when done.
    The app automatically displays your audio waveform and begins transcribing.
    Once complete, you’ll see the final text under “Transcript.”
