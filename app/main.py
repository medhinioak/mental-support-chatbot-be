from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
import requests
from pydantic import BaseModel
import numpy as np
import json
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {}

with open(os.path.join(BASE_DIR, "config_with_values.json")) as config_data:
    config = json.load(config_data)
    config_data.close()

login(token=config["HUGGINGFACES_TOKEN"])

EMAIL_ADDRESS = config["EMAIL_ADDRESS"]
MAILGUN_API_KEY=config["MAILGUN_API_KEY"]
MAILGUN_DOMAIN=config["MAILGUN_DOMAIN"]
LLM_URL = config["LLM_URL"]

origins = [
    "http://localhost:3000",  # React / Vite / Next.js dev server
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite default
]

id2label_emotions = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

id2label_suicide = {
    0: "Not on Suicide Watch",
    1: "On Suicide Watch",
}

chat = []



def decode_label_from_probs(probs, id2label):
    label_id = int(np.argmax(probs))
    return id2label.get(label_id)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load models and tokenizers
emotion_model_path = os.path.join(BASE_DIR, "models/emotion_model")
risk_model_path = os.path.join(BASE_DIR, "models/suicide_risk")
response_model_path = os.path.join(BASE_DIR, "models/response_model")

emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
emotion_model.eval()
emotion_labels = emotion_model.config.id2label

risk_tokenizer = AutoTokenizer.from_pretrained(risk_model_path)
risk_model = AutoModelForSequenceClassification.from_pretrained(risk_model_path)
risk_model.eval()
risk_labels = risk_model.config.id2label


class InputText(BaseModel):
    message: str

class EmailInput(BaseModel):
    email: str

@app.post("/chat")
def predict(input: InputText):
    text = input.message

    # Emotion prediction
    em_inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        em_logits = emotion_model(**em_inputs).logits
        em_probs = F.softmax(em_logits, dim=1)
        em_label = decode_label_from_probs(em_probs, id2label_emotions)

    print("Finished emotion detection", em_label)

    # Risk prediction
    risk_inputs = risk_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        risk_logits = risk_model(**risk_inputs).logits
        risk_probs = F.softmax(risk_logits, dim=1)
        risk_label = decode_label_from_probs(risk_probs, id2label_suicide)

    print("Finished risk detection", risk_label)
    prompt = f'''
    You are a compassionate and supportive mental health chatbot trained to respond empathetically to users. Based on the user's emotion, and suicide risk level, generate a short, empathetic response. If needed, also provide local mental health resources for users in Germany. Be warm, supportive, and never judgmental. Avoid medical advice, remind them that you're not a substitute for actual mental health professionals.

    Assume that the user will not continue the chat beyond this message so provide all relavent resources in your response. Dont prompt the user for any more inputs. Format the response such that any links or helpline numbers are easily accessible.

Format:
Detected Emotion: <emotion>
Suicide Risk Level: <On Suicide Watch | Not on Suicide Watch>

---

Few-shot examples:

Detected Emotion: sadness
Suicide Risk Level: On Suicide watch  
Response: I'm really sorry you're feeling this way. You matter, and you're not alone. It can be incredibly heavy to feel uncared for — but I want you to know that help is available. If you need someone to talk to, you can reach out to the German crisis line at 0800 111 0 111 or 0800 111 0 222 — it's free and available 24/7. You're not a burden, and you deserve support.

---

Detected Emotion: anger
Suicide Risk Level: Not on Suicide Watch  
Response: That sounds incredibly frustrating. It's okay to feel angry when things pile up like that. Your emotions are valid. Sometimes letting it out — whether by journaling, exercising, or simply talking — can help ease the pressure.

---

Detected Emotion: disappointment
Suicide Risk Level: On Suicide Watch  
Response: That sounds really tough. It takes a lot of strength to carry that weight silently. You deserve space to be heard and supported. If you're in Germany, you can call TelefonSeelsorge anonymously and for free at 0800 111 0 111. You're not alone in this, and there are people who care and want to help.

---

Detected Emotion: Sadness
Suicide Risk Level: Not On Suicide Watch  
Response: It's okay to feel down. Our emotions don't always have to make sense. What matters is being kind to yourself through it.

---

Detected Emotion: joy
Suicide Risk Level: Not on Suicide Watch
Response: That's wonderful to hear. It's so important to recognize and hold on to moments of joy — even small ones can make a big difference. Let yourself feel it fully — you deserve that lightness.

---

Detected Emotion: nervousness
Suicide Risk Level: Not on Suicide Watch
Response: Feeling nervous is completely normal — it usually means you care about what's ahead. Try to take a deep breath, one step at a time. You've handled hard things before, and you're not without strength.

---

Detected Emotion: embarrassment
Suicide Risk Level: Not on Suicide Watch
Response: Everyone feels embarrassed sometimes — it's part of being human. Try to be gentle with yourself. This moment doesn't define you, and others likely understand more than you think.

---

Now respond appropriately for the given Detected Emotion and Suicide Risk:

Detected Emotion: {em_label}  
Suicide Risk Level: {risk_label}  
    '''

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
    print("Sending request")
    response = requests.post(LLM_URL, json=payload)
    print("Got response")
    response_text = response.json()["response"]

    chat.append({
        "role": "USER",
        "message": text,
        "time": datetime.date.today()
    })
    chat.append({
        "role": "BOT",
        "message": response_text,
        "time": datetime.date.today()
    })

    return {
        "emotion": {
            "label": em_label,
            "probabilities": em_probs.tolist()
        },
        "risk": {
            "label": risk_label,
            "probabilities": risk_probs.tolist()
        },
        "response": response_text
    }

def send_email(to_email: str, body: str):
    try: 
        print(to_email)
        result = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
            "from": f"Mental Health Bot <{EMAIL_ADDRESS}>",
            "to": f"<{to_email}>",
            "subject": "Mental Health Chat History",
            "text": body
            }
        )
        print(result.text)
        return result
    except Exception as e:
        print("in exception")
        print(e)

@app.post("/export-chat")
def export_chat(input: EmailInput):
    # Format chat history into a plain text transcript
    email = input.email
    formatted = ""
    for msg in chat:
        formatted += f"{msg['role']}: {msg['message']}\nDate: {msg['time']}\n\n"


    response = send_email(
        to_email=email,
        body=formatted
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to send email")

    return {"message": f"Chat export sent to {email}"}
