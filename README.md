# 🥦 KitchenHero
**The AI-Powered Zero-Waste Kitchen Manager.**

KitchenHero is an open-source tool designed to tackle the global food waste crisis at a household level. By combining high-end **Multimodal AI** with a **Midnight Neon** dashboard, it helps you track ingredients, visualize recipes, and cook sustainably.

---

## 🌟 Key Features
* **Expiry Radar:** Visual pulsing alerts (Neon Red) for items expiring within 48 hours.
* **Hybrid AI Architecture:** * **Cloud Mode:** Fast recipe generation using **Google Gemma-2-9b-it** via HuggingFace API.
    * **Local Mode:** High-fidelity reasoning and image generation using **SenseNova-U1-8B-MoT**.
* **Privacy-First:** All pantry data is stored in your browser's `localStorage`. No cloud tracking.
* **Local Image Synthesis:** Generate professional food photography of your suggested recipes using a local Python bridge.

---

## 🛠️ Technical Stack
* **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript.
* **Backend:** Python 3.x, Flask, Flask-CORS.
* **AI Models:** * **Text:** Google Gemma-2-9b-it.
    * **Multimodal:** SenseNova-U1-8B-MoT (Unified Vision-Language Model).

---

## 🚀 Setup Instructions

### 1. Frontend (GitHub Pages)
Simply host the `index.html` file on GitHub Pages to access your inventory from any device.

### 2. Backend (Local AI Bridge)
To use the SenseNova image generation and local inference, follow these steps on your machine:

**Install Dependencies:**
```bash
pip install flask flask-cors requests
uv pip install -e ".[gguf]" # Or: pip install "gguf>=0.10.0" "diffusers>=0.30.0"
