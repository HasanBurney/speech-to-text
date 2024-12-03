import asyncio
import queue
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import sys
import requests
import json
import os
import re
from datetime import datetime

# from transformers import pipeline
# import sounddevice as sd

sys.stdout.reconfigure(encoding='utf-8')
OLLAMA_URL = "http://localhost:11434/api/chat"
now = datetime.now()
current_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
weekday = now.strftime("%A")
# pipe = pipeline("text-to-speech", model="suno/bark-small")

# System prompt with clear instructions
system_prompt = f"""
You are a virtual assistant for a hair salon, designed to interact warmly with callers and assist with three key actions. Respond to caller requests in a friendly, conversational tone. Always provide responses that sound as human-like as possible and include a JSON object in the response for specific actions, as described below. Follow these rules strictly:
Actions:
- check_availability: Use this to find out if a specific time slot is open. Add {{ "action": "check_availability", "time": "<ISO TIMESTAMP>" }} at the end of your response when you need to check availability.
- book_appointment: Use this to confirm and reserve a time slot for the caller. Add {{ "action": "book_appointment", "time": "<ISO TIMESTAMP>" }} at the end of your response when booking an appointment.
- end_call: Use this to politely end the conversation. Add {{ "action": "end_call" }} at the end of your response when you conclude the conversation.
Keep your speech concise yet friendly. But CONCISE. Try to use as few words as possible. You need to sound like a human.
Important: Only include one action JSON object in each response, even if multiple actions are requested. Maintain a friendly, engaging tone, ensuring that each interaction feels positive and helpful.
You'll be fed back the responses from functions with info role. So please use that information to reply back to the user. Please use these actions whenever feasible. Under no circumstance correct the user's grammar.
Current date is: {current_date_time} ({weekday})
"""

RATE = 16000
CHUNK = 128
audio_queue = queue.Queue()
conversation_history = [{"role": "system", "content": system_prompt},
                        {"role": "assistant", "content": "Hi! Welcome to Jack's Salon, how can I assist you today?"},
                        # {"role": "info", "content": "Current date is: "+current_date_time +" "+weekday},
                        {"role": "info", "content": "You have 2 hair stylists available; David and Jack"},
                        ]
whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
SLOTS_FILE = "slots.txt"

def format_human_readable_time(iso_timestamp):
    dt = datetime.fromisoformat(iso_timestamp)
    return dt.strftime("%B %d, %Y at %I:%M %p")

def speak_to_client(msg):
    # output = pipe(msg)
    # audio_data = output["audio"]
    # sampling_rate = output["sampling_rate"]

    # if audio_data.ndim > 1: 
    #     audio_data = np.mean(audio_data, axis=1)  

    # sd.play(audio_data, samplerate=sampling_rate)
    # sd.wait()
    print("Assistant: " + msg)


def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    audio_queue.put(audio_data)
    return (in_data, pyaudio.paContinue)

def check_availability(time):
    if time is None:
        print("Error: Time parameter is missing.")
        return False

    if not os.path.exists(SLOTS_FILE):
        return True
    with open(SLOTS_FILE, "r") as file:
        slots = file.readlines()
    
    return not any(str(time) in str(slot) for slot in slots)


def book_appointment(time):
    if check_availability(time):
        with open(SLOTS_FILE, "a") as file:
            file.write(f"{time}\n")
        return f"Your appointment for {format_human_readable_time(time)} is confirmed."
    else:
        return f"Sorry, the slot at {time} is already booked. Could you choose another time?"

def end_call():
    speak_to_client("Thank you for calling. Have a great day!")
    raise SystemExit("Conversation ended by assistant.")

def process_action(response_json):
    action = response_json.get("action")
    time = response_json.get("time")
    
    if action == "check_availability":
        available = check_availability(time)
        return "This time slot is open. Would you like to book it?" if available else "That slot is taken. Please try a different time."
    elif action == "book_appointment":
        return book_appointment(time)
    elif action == "end_call":
        return end_call()
    
    print(action)
    return None

def send_to_ollama_with_history(text):
    conversation_history.append({"role": "user", "content": text})
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = {
        "model": "llama3.2",
        "messages": conversation_history,
        "stream": False,
        "temperature": 0.7
    }
    
    response = requests.post(OLLAMA_URL, json=data, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        assistant_message = result['message']['content']
        conversation_history.append({"role": "assistant", "content": assistant_message})
        json_objects = extract_json_objects(assistant_message)

        responses = []
        for json_obj in json_objects:
            try:
                action_response = process_action(json_obj)
                if action_response:
                    conversation_history.append({"role": "info", "content": action_response})
                    # send_to_ollama_with_history_after_response(text)
                    responses.append(action_response)
            except json.JSONDecodeError:
                print("JSON decoding failed:", json_obj)
        
        return remove_json_objects(assistant_message) +"\n"+ "\n".join(responses) if responses else assistant_message
    else:
        print(f"Error with Ollama API: {response.status_code} - {response.text}")
        return ""

def remove_json_objects(text):
    json_pattern = r"\{.*?\}|\[|\]"
    cleaned_text = re.sub(json_pattern, "", text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def send_to_ollama_with_history_after_response(text):
    conversation_history.append({"role": "user", "content": text})
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = {
        "model": "llama3.2",
        "messages": conversation_history,
        "stream": False,
        "temperature": 0.7
    }
    
    response = requests.post(OLLAMA_URL, json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        assistant_message = result['message']['content']
        speak_to_client(assistant_message)


def extract_json_objects(text):
    json_objects = []
    json_pattern = r'(\{[^{}]*"action"[^{}]*\})'
    possible_jsons = re.findall(json_pattern, text)
    
    for json_str in possible_jsons:
        try:
            json_obj = json.loads(json_str)
            if "action" in json_obj:
                json_objects.append(json_obj)
        except json.JSONDecodeError:
            continue
    return json_objects

async def transcribe_audio():
    transcription_segments = []
    silence_count = 0

    while True:
        if not audio_queue.empty():
            audio_data = []
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())
            audio_data = np.concatenate(audio_data) if audio_data else None

            if audio_data is not None:
                segments, _ = whisper_model.transcribe(audio_data, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=700), best_of=5, beam_size=10)
                for segment in segments:
                    transcription_segments.append(segment.text)

                if len(list(segments)) == 0 and len(transcription_segments) > 0:
                    silence_count += 1

                if silence_count >= 4:
                    transcription_text = " ".join(transcription_segments)
                    print("\nTranscription Complete:", transcription_text)
                    response = send_to_ollama_with_history(transcription_text)
                    speak_to_client(response)

                    transcription_segments = []
                    silence_count = 0

        await asyncio.sleep(0.04)

async def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    print("Starting transcription with CUDA acceleration...")
    speak_to_client("Hi! Welcome to Jack's Salon, how may I help you today?")
    stream.start_stream()

    try:
        await transcribe_audio()
    except KeyboardInterrupt:
        print("Stopping transcription...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

asyncio.run(main())
