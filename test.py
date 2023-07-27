import json
import random
import sys
import requests
import torch
import openai
import webbrowser
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from datetime import datetime
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Timer
from threading import Lock
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

def envoyer_message(*args):
    message = entree_message.get()
    if message:
        reponse = chatbot(message)
        texte_chat.config(state='normal')  # Activez temporairement la zone de texte
        texte_chat.insert(tk.END, f"Vous : {message}\n")
        texte_chat.insert(tk.END, f"Chatbot : {reponse}\n")
        texte_chat.config(state='disabled')  # Désactivez la zone de texte après l'insertion
        entree_message.delete(0, tk.END)

# Création de la fenêtre principale et des widgets
fenetre = ThemedTk(theme="arc")  # Utilisez le thème "arc" pour un thème sombre
fenetre.title("Chatbot")

frame_chat = ttk.Frame(fenetre)
frame_chat.grid(column=0, row=0, padx=10, pady=10)

texte_chat = tk.Text(frame_chat, wrap=tk.WORD, width=50, height=20)
texte_chat.grid(column=0, row=0)

ascenseur = ttk.Scrollbar(frame_chat, command=texte_chat.yview)
ascenseur.grid(column=1, row=0, sticky=(tk.N, tk.S))
texte_chat["yscrollcommand"] = ascenseur.set

frame_entree = ttk.Frame(fenetre)
frame_entree.grid(column=0, row=1, padx=10, pady=10)

entree_message = ttk.Entry(frame_entree, width=40)
entree_message.grid(column=0, row=0)

bouton_envoyer = ttk.Button(frame_entree, text="Envoyer", command=envoyer_message)
bouton_envoyer.grid(column=1, row=0)

entree_message.bind("<Return>", envoyer_message)

fenetre.mainloop()