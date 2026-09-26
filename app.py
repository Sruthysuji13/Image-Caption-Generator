import os
import sys
import torch
import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.dataset import get_transform, Vocabulary
from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM, EncoderDecoder
import torch.serialization

st.set_page_config(
    page_title = "Image Caption Generator",
    page_icon  = "🖼️",
    layout     = "centered"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_my_model():
    if not os.path.exists(config.CHECKPOINT_PATH):
        return None, None
    torch.serialization.add_safe_globals([Vocabulary])
    checkpoint = torch.load(
        config.CHECKPOINT_PATH,
        map_location = DEVICE,
        weights_only = False
    )
    vocab   = checkpoint["vocab"]
    encoder = EncoderCNN(embed_size=config.EMBED_SIZE).to(DEVICE)
    decoder = DecoderLSTM(
        embed_size  = config.EMBED_SIZE,
        hidden_size = config.HIDDEN_SIZE,
        vocab_size  = len(vocab),
    ).to(DEVICE)
    model = EncoderDecoder(encoder, decoder).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab

@st.cache_resource
def load_blip():
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE)
        blip.eval()
        return processor, blip
    except Exception:
        return None, None

def my_model_caption(image, model, vocab):
    transform  = get_transform(train=False)
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        caption = model.generate_caption(img_tensor, vocab)
    return caption


def blip_caption(image, processor, blip_model):
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(
            **inputs,
            max_new_tokens = 50,
            num_beams      = 5,
            min_length     = 5,
        )
    return processor.decode(out[0], skip_special_tokens=True)

st.title("🖼️ Image Caption Generator")
st.markdown("Upload any image and generate captions using your own trained model or BLIP Pro.")
st.divider()

# Mode selector
mode = st.radio(
    "Choose captioning mode:",
    [
        "🧠 My Model (ResNet + LSTM + Attention)",
        "⚡ BLIP Pro (Salesforce, state of the art)",
        "🔀 Compare Both",
    ],
    horizontal=True,
)
st.divider()

# Load models based on mode
my_model, vocab = load_my_model()

if "BLIP" in mode or "Compare" in mode or "🔀" in mode:
    with st.spinner("Loading BLIP — first time takes 1-2 mins..."):
        processor, blip_model = load_blip()
else:
    processor, blip_model = None, None

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Your Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("💬 Generated Caption")

        if "My Model" in mode or "🔀" in mode:
            if my_model is None:
                st.error("⚠️ No trained model found. Run training first.")
            else:
                with st.spinner("My model is thinking..."):
                    my_cap = my_model_caption(image, my_model, vocab)

                st.markdown("**🧠 My Model (ResNet + LSTM + Attention):**")
                st.markdown(
                    f"""<div style="background:#f0f2f6;border-left:4px solid
                    #4c8bf5;border-radius:8px;padding:16px;margin-bottom:16px;">
                    <p style="font-size:17px;color:#1a1a2e;margin:0;">
                    "{my_cap}"</p></div>""",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Word count:** {len(my_cap.split())}")
                st.download_button(
                    label     = "⬇️ Download caption",
                    data      = my_cap,
                    file_name = "my_model_caption.txt",
                    mime      = "text/plain",
                    key       = "dl_my"
                )

        if "BLIP" in mode or "🔀" in mode:
            if blip_model is None:
                st.error("⚠️ Could not load BLIP. Run: pip install transformers")
            else:
                with st.spinner("BLIP is thinking..."):
                    blip_cap = blip_caption(image, processor, blip_model)

                st.markdown("**⚡ BLIP Pro (State of the Art):**")
                st.markdown(
                    f"""<div style="background:#f0f9f0;border-left:4px solid
                    #28a745;border-radius:8px;padding:16px;margin-bottom:16px;">
                    <p style="font-size:17px;color:#1a1a2e;margin:0;">
                    "{blip_cap}"</p></div>""",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Word count:** {len(blip_cap.split())}")
                st.download_button(
                    label     = "⬇️ Download caption",
                    data      = blip_cap,
                    file_name = "blip_caption.txt",
                    mime      = "text/plain",
                    key       = "dl_blip"
                )

        if "🔀" in mode and my_model is not None and blip_model is not None:
            st.divider()
            st.markdown("**📊 Comparison:**")
            c1, c2 = st.columns(2)
            c1.metric("My Model words",  len(my_cap.split()))
            c2.metric("BLIP words",      len(blip_cap.split()))
            st.markdown(
                f"""<div style="background:#fff8e1;border-left:4px solid
                #ffc107;border-radius:8px;padding:12px;margin-top:8px;">
                <p style="font-size:13px;color:#1a1a2e;margin:0;">
                💡 My model was trained from scratch on 8,091 images.<br>
                BLIP was trained on 130 million image-caption pairs.
                The gap shows exactly where more data and compute helps!
                </p></div>""",
                unsafe_allow_html=True
            )

st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;'>"
    "My Model: ResNet-50 + LSTM + Attention · Trained on Flickr8k · "
    "BLIP: Salesforce · Trained on 130M pairs"
    "</p>",
    unsafe_allow_html=True
)