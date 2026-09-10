# Rock Paper Scissors AI: Complete Deployment Guide

This project features a **Hybrid Architecture** matching both deployment models:
1. **100% Client-Side Web App (Default & Recommended):** Runs `rps_model.onnx` directly inside the user's browser using **ONNX Runtime Web (WebGL GPU / WASM)**. Deployable to **Vercel** for free with $0 hosting fees, 0 backend dependencies, and ~15ms latency.
2. **FastAPI Python Backend (Optional):** A containerized Docker server running FastAPI and ONNX Runtime with `/predict` endpoints, deployable to **Render**, **Hugging Face Spaces**, or **Google Cloud Run**.

---

## Part 1: Deploying Frontend to Vercel (100% Free & Recommended)

### Method A: Via GitHub (Recommended)

1. **Create a new repository on GitHub:**
   - Go to [github.com/new](https://github.com/new) and name it `Rock_Paper_Scissor`.
2. **Push your committed code:**
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/Rock_Paper_Scissor.git
   git push -u origin main
   ```
3. **Deploy on Vercel:**
   - Go to [vercel.com](https://vercel.com) and log in with GitHub.
   - Click **Add New... > Project** and import `Rock_Paper_Scissor`.
   - In **Project Configuration**:
     - Set **Root Directory** to: `frontend`
     - Framework Preset: `Other`
   - Click **Deploy**.
4. **Your web app is now live!** Vercel provides automatic HTTPS, enabling webcam access across all phones, tablets, and laptops.

---

### Method B: Via Vercel CLI (Direct Terminal Deploy)

```bash
cd frontend
npx vercel --prod
```
Follow the interactive prompts to link and deploy your static build.

---

## Part 2: Deploying FastAPI Backend (Optional)

The `backend/` directory contains a containerized Python service.

### Option 1: Free Hosting on Render.com (Docker)
1. Push your repository to GitHub.
2. Go to [render.com](https://render.com) and log in.
3. Click **New + > Web Service** and select `Rock_Paper_Scissor`.
4. In Settings:
   - **Root Directory:** `backend`
   - **Environment:** `Docker`
   - **Instance Type:** `Free`
5. Click **Create Web Service**.
6. Once deployed, copy your Render URL (e.g., `https://rps-backend.onrender.com`).
7. Open your frontend web app, click **⚙️ Settings**, switch to **FastAPI Python Backend**, paste your Render URL, and click **Save & Apply**!

---

### Option 2: Run Backend Locally with Docker Compose
```bash
cd backend
docker compose up --build
```
The API will be available at `http://localhost:8080`.

---

### Option 3: Run Backend Directly with Python
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

---

## Part 3: Switching Inference Engines in the Web App

Click the **⚙️ (Gear)** button in the top right header to toggle between:
* **⚡ In-Browser ONNX (WebGL GPU / WASM):** Runs inside the browser with zero server latency and total privacy.
* **🐍 Remote FastAPI Python Backend:** Routes webcam crops to your FastAPI server (`POST /predict`).

---

## Part 4: Controls & Shortcuts

| Key | Action |
|---|---|
| <kbd>Space</kbd> or <kbd>S</kbd> | Start Round (3-2-1 Countdown & Hand Capture) |
| <kbd>R</kbd> | Play Next Round / Quick Restart |
| <kbd>L</kbd> | Toggle Practice Mode (Continuous Live Gesture Recognition) |
| <kbd>M</kbd> | Toggle Sound Effects (Mute / Unmute) |
