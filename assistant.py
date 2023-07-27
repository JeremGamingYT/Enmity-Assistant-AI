import json
import random
import sys
import requests
import torch
import openai
import webbrowser
from datetime import datetime
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Timer
from threading import Lock
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Clé d'API de OpenAI (GPT-3)
openai.api_key = "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"

# Charger la base de données JSON
with open("base_de_donnees.json", "r", encoding="utf-8") as f:
    base_de_donnees = json.load(f)

# Préparer les données pour le TfidfVectorizer
phrases = []
reponses = []

rappel_lock = Lock()

for element in base_de_donnees:
    phrases.append(element["phrase"])
    reponses.append(element["reponse"])

# Créer le TfidfVectorizer et entraîner le modèle
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(phrases)

# Création des rappels
def creer_rappel(nom, date):
    rappel = {"nom": nom, "date": date}
    with open("list_rappels.json", "r") as f:
        rappels = json.load(f)
    rappels.append(rappel)
    with open("list_rappels.json", "w") as f:
        json.dump(rappels, f)
    return f"Rappel '{nom}' ajouté pour le {date}."

# Donne la date / le jour actuel
def jour_actuel():
    return datetime.now().strftime("%A %d %B")

# Intérroger l'API de GPT-2 s'il n'y a pas de réponse dans la base de donnée
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def interroger_cody_api(phrase):
    url = "https://getcody.ai/api/v1/messages"
    headers = {"Authorization": "Bearer MpF5jR45AaBYBLWFbHDAqckbDVLQN75wdibNe0p0"}
    data = {"content": phrase, "conversation_id": "306724"}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["data"]["content"]
    else:
        return "Erreur lors de l'interrogation de l'API Cody"

# Suppression des rappels
def supprimer_rappel(nom):
    with open("list_rappels.json", "r") as f:
        rappels = json.load(f)
    rappel_supprime = False
    for rappel in rappels:
        if rappel["nom"] == nom:
            rappels.remove(rappel)
            rappel_supprime = True
            break
    if rappel_supprime:
        with open("list_rappels.json", "w") as f:
            json.dump(rappels, f)
        return f"Rappel '{nom}' supprimé."
    else:
        return f"Rappel '{nom}' introuvable."

# Déclenche les rappels a la date
def declencher_rappel(nom, delai):
    def afficher_rappel():
        with rappel_lock:
            print(f"\nRappel : {nom}")
            sys.stdout.flush()

    timer = Timer(delai, afficher_rappel)
    timer.start()
    return f"Rappel '{nom}' programmé dans {delai} secondes."

# Fonctionnement de l'assistant
def chatbot(phrase_utilisateur):
    phrase_vector = vectorizer.transform([phrase_utilisateur])
    similarites = cosine_similarity(phrase_vector, X)
    index_similaire = similarites.argmax()
    
    # Ajoutez cette vérification pour éviter l'erreur KeyError
    if "action" in base_de_donnees[index_similaire]:
        action = base_de_donnees[index_similaire]["action"]
    else:
        action = None

    if action == "ajouter_rappel":
        nom = input("Quel est le nom du rappel ? ")
        date = input("Quelle est la date du rappel ? ")
        return creer_rappel(nom, date)
    else:
        return reponses[index_similaire]

while True:
    phrase_utilisateur = input("Vous : ")
    if phrase_utilisateur.lower() == "quitter":
        break
    reponse = chatbot(phrase_utilisateur)
    with rappel_lock:
        print(f"Chatbot : {reponse}")