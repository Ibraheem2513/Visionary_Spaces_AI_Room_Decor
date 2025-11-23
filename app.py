# ============================================================
#  Visionary Spaces – AI Interior Designer (HuggingFace Ready)
# ============================================================

import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from controlnet_aux import MLSDdetector
from PIL import Image
import google.generativeai as genai
import gradio as gr

# ============================================================
#   Device Auto-Detect (GPU if available, else CPU)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# Forced dtype for safety
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ============================================================
#   Load Models
# ============================================================

print("⏳ Loading Models... Please wait.")

try:
    # MLSD Edge Detector
    mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    # ControlNet Model (NO dtype arg allowed!)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-mlsd"
    ).to(device)

    # Stable Diffusion Pipeline (NO dtype arg allowed!)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        safety_checker=None
    ).to(device)

    # Scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )

    print("✅ Models Loaded Successfully!")

except Exception as e:
    print(f"❌ Critical Error loading models: {e}")


# ============================================================
#   Helper Functions
# ============================================================

def resize_keep_ratio(image, max_size=768):
    w, h = image.size
    aspect = w / h

    if w > h:
        new_w = max_size
        new_h = int(max_size / aspect)
    else:
        new_h = max_size
        new_w = int(max_size * aspect)

    # Must be divisible by 8 for SD
    new_w -= new_w % 8
    new_h -= new_h % 8

    return image.resize((new_w, new_h), Image.LANCZOS)


def reset_fields():
    return [
        None,
        "Autodetect",
        "Autodetect",
        "Autodetect",
        "",
        "",
        None,
        None
    ]


# ============================================================
#   Gemini Prompt Builder
# ============================================================

def build_gemini_prompt(api_key, image, room_type, room_style, color_palette, add_list, remove_list):

    if not api_key or api_key.strip() == "":
        raise gr.Error("🔑 Missing API Key! Please enter your Gemini API Key.")

    try:
        genai.configure(api_key=api_key)

        try:
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
        except:
            model = genai.GenerativeModel("gemini-1.5-flash")

    except Exception as e:
        raise gr.Error(f"❌ API Configuration Failed: {str(e)}")

    prefs = []
    if room_type != "Autodetect": prefs.append(f"Room Type: {room_type}")
    if room_style != "Autodetect": prefs.append(f"Style: {room_style}")
    if color_palette != "Autodetect": prefs.append(f"Color Palette: {color_palette}")

    prefs_text = " | ".join(prefs) if prefs else "Autodetect everything"

    add_string = ", ".join(add_list) if add_list else "None"
    remove_string = ", ".join(remove_list) if remove_list else "None"

    instruction = f"""
    You are an expert interior designer. Analyze the uploaded room image.
    USER PREFERENCES: {prefs_text}
    MUST ADD: {add_string}
    MUST NOT INCLUDE: {remove_string}
    OUTPUT: A single CSV string of Stable Diffusion keywords. End with "8k, photorealistic".
    """

    try:
        response = model.generate_content([instruction, image])
        text = response.text.strip()
        words = text.split()
        if len(words) > 70:
            words = words[:70]
        return " ".join(words)

    except Exception as e:
        msg = str(e)
        if "403" in msg or "API_KEY_INVALID" in msg:
            raise gr.Error("❌ Invalid Gemini API Key.")
        elif "429" in msg:
            raise gr.Error("⏳ Gemini API Quota Exceeded.")
        else:
            raise gr.Error(f"❌ Gemini Error: {msg}")


# ============================================================
#   Main Generation Logic
# ============================================================

def generate_output(image, api_key, room_type, room_style, color_palette, add_furniture, remove_furniture, use_gemini):

    if image is None:
        raise gr.Error("⚠️ No Image Uploaded!")

    if use_gemini and (not api_key or not api_key.strip()):
        raise gr.Error("🔑 Gemini Smart Prompting requires API Key.")

    try:
        image = resize_keep_ratio(image)

        add_list = [x.strip() for x in add_furniture.split(",") if x.strip()]
        remove_list = [x.strip() for x in remove_furniture.split(",") if x.strip()]

        if use_gemini:
            gr.Info("🤖 Asking Gemini for optimized prompt...")
            prompt = build_gemini_prompt(api_key, image, room_type, room_style, color_palette, add_list, remove_list)
        else:
            base_parts = []
            if room_type != "Autodetect": base_parts.append(room_type)
            if room_style != "Autodetect": base_parts.append(room_style)
            if color_palette != "Autodetect": base_parts.append(color_palette)
            if add_list: base_parts.append("with " + ", ".join(add_list))

            prompt = ", ".join(base_parts) + ", 8k, photorealistic, cinematic lighting, interior design, detailed"

        gr.Info("🎨 Generating Design...")

        structure = mlsd(
            image,
            detect_resolution=max(image.size),
            score_threshold=0.1,
            dist_threshold=0.1
        )

        neg_prompt = "text, watermark, logo, lowres, blurry, distortion, cartoon, deformed"
        if remove_list:
            neg_prompt += ", " + ", ".join(remove_list)

        result = pipe(
            prompt,
            image=structure,
            negative_prompt=neg_prompt,
            num_inference_steps=30,
            guidance_scale=9.0,
            controlnet_conditioning_scale=1.0,
            control_guidance_end=0.6
        ).images[0]

        return structure, result

    except Exception as e:
        raise gr.Error(f"❌ Generation Failed: {str(e)}")


# ============================================================
#   Modern UI – Gradio 4.x Compatible
# ============================================================

with gr.Blocks() as demo:

    # Inject CSS manually because gr.Blocks(css=...) is not allowed in Gradio 4
    gr.HTML("""
    <style>
    #title { text-align:center; font-size:42px; font-weight:800; }
    #subtitle { text-align:center; font-size:18px; color:#666; }
    </style>
    """)

    gr.Markdown("<div id='title'>🏠 Visionary Spaces</div>")
    gr.Markdown("<div id='subtitle'>AI Interior Designer with Smart Prompting</div>")

    with gr.Row():
        with gr.Column(scale=1):

            api_key_input = gr.Textbox(label="Gemini API Key", type="password",
                                       placeholder="Required when Smart Prompting is ON")

            input_img = gr.Image(type="pil", label="Upload Empty Room", height=260)

            with gr.Accordion("Advanced Preferences", open=True):
                room_type_in = gr.Dropdown(
                    ["Autodetect", "Living Room", "Bedroom", "Kitchen", "Office"],
                    label="Room Type",
                    value="Autodetect"
                )
                room_style_in = gr.Dropdown(
                    ["Autodetect", "Modern", "Minimalist", "Luxury", "Bohemian"],
                    label="Interior Style",
                    value="Autodetect"
                )
                color_in = gr.Dropdown(
                    ["Autodetect", "White & Black", "Beige & Green", "Blue & White"],
                    label="Color Palette",
                    value="Autodetect"
                )

            add_furn_in = gr.Textbox(label="Add Furniture", placeholder="sofa, plants")
            remove_furn_in = gr.Textbox(label="Do NOT Add", placeholder="TV, carpets")

            use_gemini_chk = gr.Checkbox(label="Use Gemini Smart Prompting", value=True)

            with gr.Row():
                reset_btn = gr.Button("Reset")
                gen_btn = gr.Button("Generate Design", variant="primary")

        with gr.Column(scale=2):
            structure_out = gr.Image(label="Structure Map", height=300)
            final_out = gr.Image(label="Final Design", height=300)

    gen_btn.click(
        generate_output,
        inputs=[
            input_img, api_key_input, room_type_in, room_style_in,
            color_in, add_furn_in, remove_furn_in, use_gemini_chk
        ],
        outputs=[structure_out, final_out]
    )

    reset_btn.click(
        reset_fields,
        outputs=[
            input_img, room_type_in, room_style_in, color_in,
            add_furn_in, remove_furn_in, structure_out, final_out
        ]
    )

demo.launch(share=True)
