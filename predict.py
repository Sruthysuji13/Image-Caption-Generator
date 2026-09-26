import sys
import os
import argparse

import torch
from PIL import Image

import config
from src.dataset import get_transform
from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM, EncoderDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    """Load model and vocab from a saved checkpoint file."""
    print(f"[predict] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Rebuild vocab from checkpoint
    vocab      = checkpoint["vocab"]
    vocab_size = len(vocab)

    # Rebuild model architecture and load saved weights
    encoder = EncoderCNN(embed_size=config.EMBED_SIZE).to(DEVICE)
    decoder = DecoderLSTM(
        embed_size  = config.EMBED_SIZE,
        hidden_size = config.HIDDEN_SIZE,
        vocab_size  = vocab_size,
    ).to(DEVICE)
    model = EncoderDecoder(encoder, decoder).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"[predict] Model loaded — vocab size: {vocab_size} | "
          f"trained for {checkpoint['epoch']} epochs | "
          f"best val loss: {checkpoint['val_loss']:.4f}")
    return model, vocab


def predict(image_path, model, vocab):
    """
    Given a path to any image, return a generated caption string.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Preprocess exactly like validation images
    transform = get_transform(train=False)
    image     = Image.open(image_path).convert("RGB")
    image     = transform(image).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    with torch.no_grad():
        caption = model.generate_caption(image, vocab)

    return caption


def main():
    parser = argparse.ArgumentParser(description="Image Caption Generator")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file you want to caption",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.CHECKPOINT_PATH,
        help="Path to model checkpoint (default: best_model.pth in root)",
    )
    args = parser.parse_args()

    model, vocab = load_model(args.checkpoint)
    caption      = predict(args.image, model, vocab)

    print(f"\n── Caption ───────────────────────────────")
    print(f"  {caption}")
    print(f"──────────────────────────────────────────\n")


if __name__ == "__main__":
    main()