import whisper
import warnings
import sounddevice as sd
import numpy as np
import subprocess
import os
from google import genai

warnings.filterwarnings("ignore")

# AQUÍ PEGÁS TU CLAVE EXACTA (la que empieza por AQ.)
os.environ["GEMINI_API_KEY"] = "TU_API_KEY_AQUI"

client = genai.Client()

def hablar(texto):
    print(f"AVAD dice: {texto}")
    subprocess.run(["say", "-v", "Monica", texto])

print("Cargando el modelo de Whisper (AVAD)...")
model = whisper.load_model("tiny")
hablar("Sistema listo con inteligencia de Gemini.")

print("\n--- AVAD LISTO PARA ESCUCHAR ---")
input("Presiona la tecla ENTER en tu teclado y haz tu pregunta...")

print("Grabando 6 segundos... ¡Habla ahora!")
duration = 6  
fs = 16000  
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()  

audio = audio.flatten()

print("Procesando tu voz con Whisper...")
result = model.transcribe(audio, language="es")
pregunta_usuario = result["text"].strip()

print(f"\n---> Pregunta capturada: '{pregunta_usuario}' <---")

if pregunta_usuario:
    hablar("Déjame pensar la respuesta...")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Eres AVAD, un asistente virtual de análisis de datos e inteligencia artificial. Responde de forma directa, clara, concisa y en español, ideal para ser leída en voz alta en máximo dos oraciones a la siguiente pregunta: {pregunta_usuario}",
        )
        
        respuesta_ia = response.text.strip()
        print(f"\nRespuesta de Gemini: {respuesta_ia}")
        hablar(respuesta_ia)
        
    except Exception as e:
        print(f"Error conectando con Gemini: {e}")
        hablar("Tuve un problema al consultar mi inteligencia.")
else:
    hablar("No logré escuchar tu pregunta. Inténtalo de nuevo.")