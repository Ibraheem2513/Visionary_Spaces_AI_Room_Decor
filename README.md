# Visionary Spaces – AI Interior Designer

**Visionary Spaces** is a modern AI-powered interior design tool that generates photorealistic room designs from an uploaded empty room image. It integrates **Stable Diffusion** with **ControlNet** for structure-aware generation and optionally uses **Gemini AI** for smart prompt optimization.

---

## Features

- Upload an image of an empty room and generate realistic interior designs.
- Control room type, interior style, and color palette.
- Add or remove furniture and objects from the design.
- Gemini AI smart prompting for optimized Stable Diffusion keywords.
- GPU and CPU compatible (automatically detects device).
- Minimalistic, user-friendly **Gradio** interface.
- High-resolution photorealistic outputs with cinematic lighting.

---

## Demo

- Checkout the demo video here:
https://drive.google.com/file/d/1g8UiEEPLl2lcR5vA6vkB0uQCBjXN_qVO/view?usp=sharing

---

## Requirements

```text
torch
transformers
diffusers
accelerate
gradio
pillow
controlnet_aux
mediapipe
google.generativeai
numpy==1.26.4
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Setup & Usage

1. Clone the repository:

```bash
git clone https://github.com/Ibraheem2513/Visionary_Spaces_AI_Room_Decor
cd Visionary_Spaces_AI_Room_Decor
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Upload an image and adjust preferences in the Gradio interface.

5. Optional: Enable **Gemini Smart Prompting** and provide your Gemini API key.

---

## Files

- `app.py` – Main Python script containing the app logic.
- `requirements.txt` – Python dependencies.
- `.gitignore` – Files/folders to ignore in Git.
- `README.md` – Project documentation.

---

## Notes

- Works on CPU but GPU is recommended for faster generation.
- ControlNet uses MLSD edge detection for structure-guided generation.
- Gemini AI API key is required for smart prompt generation.
- Generated images are 768px by default (modifiable in the code).

---

## License

MIT License © 2025 Muhammad Ibraheem
