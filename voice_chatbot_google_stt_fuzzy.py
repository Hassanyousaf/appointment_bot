from flask import Flask, request, jsonify
import sqlite3
import json
import os
import re
import requests
from datetime import datetime
from dateutil import parser as dateparser
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Hugging Face API configuration
HUGGINGFACE_API_TOKEN = "key"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/ruslanmv/Medical-Llama3-Chatbot"

# SQLite database setup
def init_db():
    conn = sqlite3.connect("appointments.db")
    c = conn.cursor()
    
    # Check if table exists and has the right columns
    c.execute("PRAGMA table_info(appointments)")
    columns = c.fetchall()
    column_names = [col[1] for col in columns]
    
    if 'appointments' not in [table[0] for table in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
        c.execute("""CREATE TABLE appointments (
            id INTEGER PRIMARY KEY,
            patient_name TEXT,
            contact TEXT,
            doctor TEXT,
            time TEXT,
            symptoms TEXT
        )""")
    elif 'symptoms' not in column_names:
        c.execute("ALTER TABLE appointments ADD COLUMN symptoms TEXT")
    
    conn.commit()
    conn.close()
init_db()

# Doctor availability (in production this would come from a database)
DOCTOR_AVAILABILITY = {
    "Dr. Smith": {
        "specialty": "Cardiology",
        "slots": [
            "2025-06-25 09:00",
            "2025-06-25 11:00",
            "2025-06-26 14:00",
            "2025-06-26 16:00"
        ]
    },
    "Dr. Johnson": {
        "specialty": "Pediatrics",
        "slots": [
            "2025-06-25 10:00",
            "2025-06-26 09:00",
            "2025-06-27 13:00"
        ]
    }
}

# Improved LLM call with error handling
def query_llm(prompt):
    symptom_responses = {
        "fever" :"Based on your symptoms, I recommend seeing a general physician",
        "pain":"For your pain symptoms, I suggest visiting Dr. Johnson for checkup",
        "cough":"A persistent cough might need attention from a pulmonologist",
        "headache": "For headaches, you could consult with Dr. Smith"
    }

    # Hugging face API if available
    try:
        payload = {"inputs" : prompt}
        response = requests.post(
            API_URL,
            headers= headers,
            json = payload,
            timeout=10 # 10 sec timeout
        )
        response.raise_for_status()

        return response.json()[0]["generated_text"]
    except Exception as e:
        print(f"LLM Error:{e}")
        # falling back to simple symptom matching
        for symptom in symptom_responses:
            if symptom in prompt.lower():
                return symptom_responses[symptom]
            return "I recommend consulting with a doctor about your symptoms. Would you like me to book an appointment?"
'''
def query_llm(prompt):
    payload = {"inputs": prompt, "parameters": {"max_length": 200}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "I didn't understand that. Could you please rephrase?")
        elif isinstance(data, dict):
            return data.get("generated_text", "I didn't understand that. Could you please rephrase?")
        else:
            return "I'm having trouble processing your request. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Sorry, I'm experiencing technical difficulties. Please try again later. ({str(e)})"
'''
# Speech to Text via Google STT
def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    
    # Convert to WAV format for better compatibility
    if audio_file_path.endswith('.webm'):
        audio = AudioSegment.from_file(audio_file_path, format="webm")
        wav_path = "static/processed.wav"
        audio.export(wav_path, format="wav")
    else:
        wav_path = audio_file_path

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return "Speech recognition service unavailable"

# TTS with error handling
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en")
        output_path = "static/output.mp3"
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""

# Improved slot matching logic
def match_appointment_slot(user_input, slots):
    # Try to parse exact date/time
    try:
        user_time = dateparser.parse(user_input, fuzzy=True)
        if user_time:
            for slot in slots:
                slot_time = datetime.strptime(slot, "%Y-%m-%d %H:%M")
                if abs((slot_time - user_time).total_seconds()) < 1800:  # 30 min window
                    return slot
    except:
        pass

    # Handle relative time expressions
    normalized_input = user_input.lower()
    
    # Match "first", "second", etc.
    ordinal_map = {"first": 0, "second": 1, "third": 2, "fourth": 3, "last": -1}
    for term, index in ordinal_map.items():
        if term in normalized_input:
            try:
                return slots[index] if index >= 0 else slots[index]
            except IndexError:
                return None

    # Match time patterns like "9 am", "3 pm"
    time_pattern = r'(\d{1,2})\s*(am|pm)?'
    match = re.search(time_pattern, normalized_input)
    if match:
        hour = int(match.group(1))
        period = match.group(2) or ""
        
        # Convert to 24-hour format
        if 'pm' in period and hour < 12:
            hour += 12
        if 'am' in period and hour == 12:
            hour = 0
            
        # Find closest matching slot
        best_match = None
        min_diff = float('inf')
        
        for slot in slots:
            slot_time = datetime.strptime(slot, "%Y-%m-%d %H:%M")
            hour_diff = abs(slot_time.hour - hour)
            
            if hour_diff < min_diff:
                min_diff = hour_diff
                best_match = slot
                
        return best_match

    return None

# Format slots for display
def format_slots(slots):
    formatted = []
    for i, slot in enumerate(slots):
        dt = datetime.strptime(slot, "%Y-%m-%d %H:%M")
        formatted.append(f"{i+1}. {dt.strftime('%A, %B %d at %I:%M %p')}")
    return "\n".join(formatted)

# Main conversation logic
def process_query(text, context):
    text = text.lower().strip()
    
    # Reset context if requested
    if any(word in text for word in ["restart", "reset", "start over", "new session"]):
        return "Okay, let's start over. How can I help you today?", {}

    # Initial state - collect symptoms or booking intent
    if not context:
        if any(word in text for word in ["book", "appointment", "schedule"]):
            context["intent"] = "booking"
            doctors_list = ", ".join([f"{doctor} ({details['specialty']})" 
                                   for doctor, details in DOCTOR_AVAILABILITY.items()])
            return "Sure, I can help with appointments. Which doctor would you like to see? We have: " + doctors_list, context
        
        context["intent"] = "symptoms"
        return "Please describe your symptoms so I can recommend the right doctor.", context
    
    # Symptom collection flow
    if context.get("intent") == "symptoms" and not context.get("symptoms"):
        context["symptoms"] = text
        response = query_llm(f"Patient reports: {text}. Which specialist should they see?")
        
        # Find the most appropriate doctor based on response
        recommended_doctor = None
        for doctor, details in DOCTOR_AVAILABILITY.items():
            if details["specialty"].lower() in response.lower():
                recommended_doctor = doctor
                break
        
        if recommended_doctor:
            context["doctor"] = recommended_doctor
            context["slots"] = DOCTOR_AVAILABILITY[recommended_doctor]["slots"]
            slots_formatted = format_slots(context["slots"])
            return (f"{response}\n\nAvailable slots with Dr. {recommended_doctor}:\n"
                   f"{slots_formatted}\nWould you like to book an appointment?"), context
        else:
            return f"{response}\n\nWould you like me to book you with a general physician?", context
        
    # Doctor selection
    if not context.get("doctor"):
        # Match doctor by name or specialty
        selected_doctor = None
        for doctor, details in DOCTOR_AVAILABILITY.items():
            if doctor.lower() in text or details["specialty"].lower() in text:
                selected_doctor = doctor
                break
        
        if selected_doctor:
            context["doctor"] = selected_doctor
            context["slots"] = DOCTOR_AVAILABILITY[selected_doctor]["slots"]
            slots_formatted = format_slots(context["slots"])
            return f"Great! Available slots for Dr. {selected_doctor}:\n{slots_formatted}\nPlease choose a time or say 'first available'.", context
        else:
            doctors_list = ", ".join([f"{doctor} ({details['specialty']})" 
                                   for doctor, details in DOCTOR_AVAILABILITY.items()])
            return "I didn't recognize that doctor. Please choose from: " + doctors_list, context
    
    # Time selection
    if context.get("doctor") and not context.get("time"):
        slot = match_appointment_slot(text, context["slots"])
        
        if slot:
            context["time"] = slot
            return "Got it! What's your full name?", context
        else:
            slots_formatted = format_slots(context["slots"])
            return f"Sorry, I didn't understand that time. Available slots:\n{slots_formatted}\nYou can say 'first slot' or 'June 22 at 4 pm'.", context
    
    # Patient name collection
    if context.get("time") and not context.get("patient_name"):
        if len(text.split()) >= 2:  # At least first and last name
            context["patient_name"] = text.title()
            return "And your phone number? Please include area code.", context
        else:
            return "Please provide your full name (first and last).", context
    
    # Phone number collection
    if context.get("patient_name") and not context.get("contact"):
        # Simple phone number validation - at least 7 digits
        digits = [c for c in text if c.isdigit()]
        if len(digits) >= 7:
            context["contact"] = ''.join(digits)
            
            # Book the appointment
            conn = sqlite3.connect("appointments.db")
            c = conn.cursor()
            try:
                c.execute("""INSERT INTO appointments 
                           (patient_name, contact, doctor, time, symptoms) 
                           VALUES (?, ?, ?, ?, ?)""",
                          (context["patient_name"], 
                           context["contact"], 
                           context["doctor"], 
                           context["time"],
                           context.get("symptoms", "N/A")))
                conn.commit()
                
                # Format confirmation
                dt = datetime.strptime(context["time"], "%Y-%m-%d %H:%M")
                confirmation = (
                    f"✅ Appointment confirmed with Dr. {context['doctor']}!\n"
                    f"• Date/Time: {dt.strftime('%A, %B %d at %I:%M %p')}\n"
                    f"• Patient: {context['patient_name']}\n"
                    f"• Contact: {context['contact']}\n"
                    "A confirmation will be sent to your phone."
                )
                return confirmation, {}  # Empty context to start fresh
            except sqlite3.Error as e:
                return f"Sorry, there was an error booking your appointment: {str(e)}", context
            finally:
                conn.close()
        else:
            return "That doesn't look like a complete phone number. Please provide at least 7 digits including area code.", context
    
    # Fallback response
    return "I'm not sure I understand. Could you please rephrase or say 'restart' to begin again?", context

# Routes
@app.route("/chat", methods=["POST"])
def chat():
    # Handle audio input
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    input_path = "static/input_audio.webm"
    audio_file.save(input_path)
    
    text = speech_to_text(input_path)
    context = json.loads(request.form.get("context", "{}"))
    
    # Only process if we got valid text input
    if text.strip():
        response_text, new_context = process_query(text, context)
    else:
        response_text = "I didn't catch that. Could you please repeat?"
        new_context = context
    
    # Generate unique filename for each response to prevent caching
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = f"static/output_{timestamp}.mp3"
    
    # Always generate new audio for each response
    try:
        tts = gTTS(text=response_text, lang="en")
        tts.save(output_path)
        audio_url = f"/{output_path}"
    except Exception as e:
        print(f"TTS Error: {e}")
        audio_url = ""
    
    # Clean up old audio files (keep last 5)
    audio_files = sorted([f for f in os.listdir("static") if f.startswith("output_") and f.endswith(".mp3")])
    for old_file in audio_files[:-5]:
        try:
            os.remove(os.path.join("static", old_file))
        except:
            pass
    
    return jsonify({
        "transcript": text,
        "response": response_text,
        "audio": audio_url,
        "context": new_context
    })

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)