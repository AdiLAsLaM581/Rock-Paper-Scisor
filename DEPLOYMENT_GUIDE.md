# Deployment Guide: Rock Paper Scissors Client-Side AI Web App

This project has been migrated to a **100% Client-Side Web Application** powered by **ONNX Runtime Web**. The deep learning model (`rps_model.onnx`, 13.9 MB) runs directly inside the user's browser using **WebGL (GPU)** or **WebAssembly (CPU)**.

---

## Key Advantages
- **Zero Backend Required:** No Google Cloud Run, no Docker, no Render, no credit card required.
- **100% Free Hosting:** Can be hosted permanently on **Vercel**, **Netlify**, or **GitHub Pages** for $0.
- **Complete Privacy:** Camera video frames are processed entirely in browser memory; no video or image data is ever sent to any server.
- **Ultra-Low Latency:** In-browser inference runs in ~10–25ms per frame.

---

## 1. Local Testing

To test the application locally on your computer:

1. Open your terminal (PowerShell or Command Prompt) in the project directory.
2. Start Python's built-in HTTP server:
   ```bash
   python -m http.server 3000 --directory frontend
   ```
3. Open your browser and navigate to:
   **[http://localhost:3000](http://localhost:3000)**
4. Allow camera permissions when prompted.
5. Watch the **"Streaming Deep Learning Model"** progress bar download and initialize the WebGL GPU engine.
6. Press **Spacebar** or click **START ROUND** to play against the AI!

---

## 2. Deploy to Vercel (100% Free)

### Option A: Via GitHub (Recommended)

1. **Commit and Push your project to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy client-side Rock Paper Scissors AI to Vercel"
   git push
   ```

2. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com) and log in with GitHub.
   - Click **Add New... > Project**.
   - Select your `Rock_Paper_Scissor` repository.

3. **Configure the Project:**
   - In the **Project Configuration** screen:
     - Set **Root Directory** to: `frontend`
     - Framework Preset: `Other` (Static HTML/JS)
   - Click **Deploy**.

4. **Done!**
   - Vercel will deploy your application to an HTTPS URL (e.g., `https://rock-paper-scissors-ai.vercel.app`) in ~30 seconds.
   - Because Vercel provides free HTTPS, your browser webcam permissions will work seamlessly on all desktop and mobile devices.

---

### Option B: Via Vercel CLI (Direct Terminal Deploy)

If you don't want to use GitHub:

1. In your terminal, navigate to the `frontend` folder:
   ```bash
   cd "c:\Users\HP\Desktop\AI Project\Rock_Paper_Scissor\frontend"
   ```

2. Run Vercel CLI:
   ```bash
   npx vercel --prod
   ```

3. Follow the quick terminal prompts (press Enter for defaults). Vercel will upload the static files and output your live production URL.

---

## 3. Keyboard Shortcuts & Controls

| Key | Action |
|---|---|
| <kbd>Space</kbd> or <kbd>S</kbd> | Start Round (3-2-1 Countdown & Capture) |
| <kbd>R</kbd> | Play Again / Restart Round |
| <kbd>L</kbd> | Toggle Practice Mode (Continuous Live AI Recognition) |
| <kbd>M</kbd> | Toggle Synthesized Sound Effects (Mute / Unmute) |

---

## 4. Technical Architecture

- **Engine:** `onnxruntime-web` (v1.20.1) via CDN.
- **Hardware Acceleration:** WebGL GPU execution provider with automatic WASM fallback.
- **Model:** `rps_model.onnx` (13.9 MB, 150x150 RGB input tensor, 3 classes: Paper, Rock, Scissors).
- **Audio:** Web Audio API synth oscillators (zero external audio asset files needed).
- **Caching:** `vercel.json` serves `rps_model.onnx` with `max-age=31536000, immutable` headers so users only download the model once and cache it locally in their browser.
