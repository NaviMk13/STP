import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import random

# Seiteneinstellungen für Darkmode & Design
st.set_page_config(page_title="AI Schere-Stein-Papier", layout="wide")

# Custom CSS für echtes Arcade-Feeling
st.markdown("""
    <style>
    .main { background-color: #1a1a1a; color: white; }
    stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Modell laden
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = load_my_model()

# Session State Initialisierung
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'highscore' not in st.session_state:
    st.session_state.highscore = 0

# Sidebar für Menü und Stats
st.sidebar.title("🎮 Game Menu")
st.sidebar.subheader(f"🔥 Streak: {st.session_state.streak}")
st.sidebar.subheader(f"🏆 Highscore: {st.session_state.highscore}")
st.sidebar.divider()
stroke_width = st.sidebar.slider("Stiftdicke anpassen:", 5, 30, 15)
if st.sidebar.button("Statistiken zurücksetzen"):
    st.session_state.streak = 0
    st.rerun()

st.title("Schere, Stein, Papier vs. KI")
st.write("Zeichne dein Symbol auf das Canvas und fordere die KI heraus!")

col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if st.button("SPIELEN!"):
        if canvas_result.image_data is not None:
            # Countdown-Animation
            countdown_placeholder = st.empty()
            for i in ["3...", "2...", "1...", "GO!"]:
                countdown_placeholder.header(f"⏳ {i}")
                time.sleep(0.6)
            countdown_placeholder.empty()

            # Bildvorverarbeitung (RGBA -> Graustufen -> S/W Kontrast)
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            img = img.convert('L') # Zu Graustufen
            img = ImageOps.invert(img) # Invertieren, falls Modell Schwarz auf Weiß erwartet (oder umgekehrt)
            img = img.resize((224, 224))
            
            # KI-Vorhersage
            img_array = np.asarray(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(np.stack((img_array,)*3, axis=-1), axis=0) # Zu RGB Format für Keras
            
            prediction = model.predict(img_array)
            index = np.argmax(prediction)
            user_choice = labels[index].split(" ")[1] # Extrahiert "Papier", "Schere" oder "Stein" 
            conf_score = prediction[0][index]

            # Computer Wahl
            comp_options = ["Papier", "Schere", "Stein"]
            comp_choice = random.choice(comp_options)

            # Ergebnis-Logik
            st.subheader(f"Du hast gezeichnet: **{user_choice}** ({conf_score:.1%})")
            st.subheader(f"Computer wählt: **{comp_choice}**")
            
            if user_choice == comp_choice:
                st.info("Unentschieden! 🤝")
            elif (user_choice == "Schere" and comp_choice == "Papier") or \
                 (user_choice == "Stein" and comp_choice == "Schere") or \
                 (user_choice == "Papier" and comp_choice == "Stein"):
                st.success("DU GEWINNST! 🎉")
                st.session_state.streak += 1
                if st.session_state.streak > st.session_state.highscore:
                    st.session_state.highscore = st.session_state.streak
                    st.balloons()
            else:
                st.error("KI gewinnt! 🤖")
                st.session_state.streak = 0
            
            st.image(img, caption="KI-Sicht (S/W Vorschau)", width=150)
