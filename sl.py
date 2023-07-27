import streamlit as st
from streamlit_chat import message
import json
import random
import sys
import requests
import torch
import openai
import webbrowser
import numpy as np
#import tkinter as tk
#from tkinter import ttk
from datetime import datetime
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Timer
from threading import Lock

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
            st.write(f"\nRappel : {nom}")
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


st.title("Bard AI - By JeremGaming")
# V4
# Initialisez l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Écrivez quelque chose..")

if user_input:
    st.session_state.messages.append({"content": f"User : {user_input}", "is_user": True})
    response = chatbot(user_input)
    st.session_state.messages.append({"content": f"Bard : {response}", "is_user": False})

# Affichez l'historique des messages
for msg in st.session_state.messages:
    message(msg["content"], is_user=msg["is_user"])


# V3
#user_input = st.chat_input("Écrivez quelque chose..")

#if user_input:
#    message(f"User : {user_input}", is_user=True)
#    response = chatbot(user_input)
#    message(f"Bard : {response}", is_user=False)


# V2
#user_input = st.chat_input("Écrivez quelque chose..")
#bard_response = st.chat_message("user")

#if user_input:
#    response = chatbot(user_input)
#    with bard_response:
#        st.write(f"Bard : {response}")


#V1
#user_input = st.chat_input("Écrivez quelque chose..")
#if user_input:
#    response = chatbot(user_input)
#    st.write(f"Bard : {response}")