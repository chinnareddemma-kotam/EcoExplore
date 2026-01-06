import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import sqlite3
import hashlib
import pandas as pd
import folium
from streamlit_folium import st_folium
from soildetection.model import build_model
import torchvision.models.segmentation as segmentation
import numpy as np

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="EcoExplore",
    page_icon="🌱",
    layout="wide"
)

# -------------------
# Custom CSS (UI)
# -------------------
st.markdown("""
<style>
.main { background-color: #f6f8fa; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f3d2e, #145a32);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.title {
    font-size: 36px;
    font-weight: 700;
    color: #145a32;
}

.subtitle {
    font-size: 18px;
    color: #555;
}

.stButton>button {
    border-radius: 10px;
    padding: 10px 20px;
    background-color: #145a32;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# -------------------
# Paths
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOIL_MODEL_PATH = os.path.join(BASE_DIR, "soildetection", "models", "best.pth")
VEG_MODEL_PATH = os.path.join(BASE_DIR, "vegetationsegmentation", "models", "vegetation_model.pth")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Session State
# -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "menu" not in st.session_state:
    st.session_state.menu = "Login"

# -------------------
# Database
# -------------------
DB_NAME = "users.db"

def get_db():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_tables():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            model_type TEXT,
            prediction TEXT,
            confidence REAL
        )
    """)
    conn.commit()

create_tables()

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def register_user(u, p):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username,password) VALUES (?,?)",
                (u, hash_password(p)))
    conn.commit()

def login_user(u, p):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?",
                (u, hash_password(p)))
    return cur.fetchone()

def save_history(user, model, pred, conf):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO history VALUES (NULL,?,?,?,?)",
                (user, model, pred, conf))
    conn.commit()

def get_history(user):
    return pd.read_sql("SELECT * FROM history WHERE username=?",
                       get_db(), params=(user,))

# -------------------
# Load Models
# -------------------
@st.cache_resource
def load_soil_model():
    ckpt = torch.load(SOIL_MODEL_PATH, map_location=device)
    model = build_model(
        model_name="resnet50",
        num_classes=len(ckpt["class_names"]),
        pretrained=False
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt["class_names"]

@st.cache_resource
def load_veg_model():
    ckpt = torch.load(VEG_MODEL_PATH, map_location=device)
    model = segmentation.deeplabv3_resnet50(weights=None, num_classes=2)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# -------------------
# Sidebar
# -------------------
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)

st.sidebar.title("🌱 EcoExplore")

menu = st.sidebar.radio(
    "Navigation",
    ["Login", "Home", "History", "Map", "Logout"]
    if st.session_state.logged_in else ["Login"],
    index=["Login", "Home", "History", "Map", "Logout"].index(st.session_state.menu)
)

# -------------------
# LOGIN
# -------------------
if menu == "Login":
    st.session_state.menu = "Login"

    st.markdown("""
    <div class="card">
        <h2>🔐 EcoExplore Login</h2>
        <p>Access AI-powered archaeological analysis tools</p>
    </div>
    """, unsafe_allow_html=True)

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    c1, c2 = st.columns(2)
    if c1.button("Login"):
        if login_user(u, p):
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.menu = "Home"
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    if c2.button("Register"):
        try:
            register_user(u, p)
            st.success("Account created")
        except:
            st.error("Username already exists")

# -------------------
# HOME
# -------------------
elif menu == "Home":
    st.session_state.menu = "Home"

    st.markdown("""
    <div class="card">
        <div class="title"> Archaeological Site Mapping</div>
        <div class="subtitle">
            AI-Driven Archaeological Intelligence Platform for Soil Detection and Vegetation Segmentation.
        </div>
        <p>
        EcoExplore combines soil detection and vegetation segmentation using
        deep learning to support archaeological exploration and site prioritization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📌 What This App Does</h3>
        <ul>
            <li>AI-based soil type detection from field images</li>
            <li>Vegetation segmentation from satellite imagery</li>
            <li>Vegetation coverage percentage estimation</li>
            <li>User login and analysis history tracking</li>
            <li>Interactive satellite map visualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Interactive Analysis</h3>
        <p>Upload images below to run AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # -------- Soil --------
    with col1:
        st.markdown("<div class='card'><h3>🌱 Soil Detection</h3></div>", unsafe_allow_html=True)
        f = st.file_uploader("Upload Soil Image", type=["jpg", "png"], key="soil")

        if f:
            img = Image.open(f).convert("RGB")
            st.image(img, use_container_width=True)

            model, classes = load_soil_model()
            tf = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            x = tf(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(x)
                pred = torch.argmax(out, 1).item()
                conf = torch.softmax(out, 1)[0][pred].item() * 100

            st.success(f"Soil Type: **{classes[pred]}**")
            st.info(f"Confidence: {conf:.2f}%")

            save_history(st.session_state.username,
                         "Soil Detection",
                         classes[pred],
                         conf)

    # -------- Vegetation --------
    with col2:
        st.markdown("<div class='card'><h3>🌿 Vegetation Segmentation</h3></div>", unsafe_allow_html=True)
        f2 = st.file_uploader("Upload Satellite Image", type=["jpg", "png"], key="veg")

        if f2:
            img = Image.open(f2).convert("RGB")
            st.image(img, use_container_width=True)

            veg_model = load_veg_model()
            tf = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

            x = tf(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = veg_model(x)["out"]
                mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            veg_pixels = np.sum(mask == 1)
            total_pixels = mask.size
            veg_percent = (veg_pixels / total_pixels) * 100

            st.success(f"Vegetation Coverage: {veg_percent:.2f}%")

            save_history(st.session_state.username,
                         "Vegetation Segmentation",
                         "Vegetation Detected",
                         veg_percent)

# -------------------
# HISTORY
# -------------------
elif menu == "History":
    st.session_state.menu = "History"

    st.markdown("""
    <div class="card">
        <h3>📊 Prediction History</h3>
        <p>Your previous AI analysis records</p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(get_history(st.session_state.username), use_container_width=True)

# -------------------
# MAP
# -------------------
elif menu == "Map":
    st.session_state.menu = "Map"

    st.markdown("""
    <div class="card">
        <h3>🗺️ Satellite Map View</h3>
        <p>Explore geographical regions using satellite imagery</p>
    </div>
    """, unsafe_allow_html=True)

    m = folium.Map(location=[20.59, 78.96], zoom_start=5)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri"
    ).add_to(m)

    st_folium(m, height=500)

# -------------------
# LOGOUT
# -------------------
elif menu == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.menu = "Login"
    st.success("Logged out")
    st.rerun()
