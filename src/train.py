import os, sys, math, time
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace('src', ''))
import config
from src.dataset import get_loaders
from models.encoder import EncoderCNN
from models.decoder import DecoderLSTM, EncoderDecoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] Using: {DEVICE}')


def evaluate_bleu(model, loader, vocab):
    model.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for images, captions in loader:
            images = images.to(DEVICE)
            for i in range(images.size(0)):
                caption_str = model.generate_caption(images[i].unsqueeze(0), vocab)
                hypothesis  = caption_str.split()
                ref_str     = vocab.decode(captions[i].tolist())
                hypotheses.append(hypothesis)
                references.append([ref_str.split()])
    smoother = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25), smoothing_function=smoother)
    return bleu1, bleu4


def train():
    print('\n[Setup] Loading data...')
    train_loader, val_loader, test_loader, vocab = get_loaders()
    vocab_size = len(vocab)

    encoder = EncoderCNN(embed_size=config.EMBED_SIZE).to(DEVICE)
    decoder = DecoderLSTM(config.EMBED_SIZE, config.HIDDEN_SIZE, vocab_size).to(DEVICE)
    model   = EncoderDecoder(encoder, decoder).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = math.inf
    print(f'\n[Train] Starting training for {config.NUM_EPOCHS} epochs...\n')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, captions) in enumerate(train_loader):
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            logits  = model(images, captions)
            targets = captions[:, 1:]
            min_len = min(logits.size(1), targets.size(1))
            logits  = logits[:, :min_len, :]
            targets = targets[:, :min_len]
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f'  Epoch [{epoch}/{config.NUM_EPOCHS}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(DEVICE), captions.to(DEVICE)
                logits  = model(images, captions)
                targets = captions[:, 1:]
                min_len = min(logits.size(1), targets.size(1))
                logits  = logits[:, :min_len, :]
                targets = targets[:, :min_len]
                loss    = criterion(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
)
                val_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        elapsed   = time.time() - start_time
        print(f'\nEpoch [{epoch}/{config.NUM_EPOCHS}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Time: {elapsed:.1f}s')

        if epoch % 5 == 0:
            bleu1, bleu4 = evaluate_bleu(model, val_loader, vocab)
            print(f'  BLEU-1: {bleu1:.4f} | BLEU-4: {bleu4:.4f}')

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss'   : best_val_loss,
                'vocab'      : vocab,
            }, config.CHECKPOINT_PATH)
            print(f'  ✓ Saved best model (val_loss={best_val_loss:.4f})')

        scheduler.step(avg_val)
        print()

    print('[Eval] Loading best model for test evaluation...')
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    bleu1, bleu4 = evaluate_bleu(model, test_loader, vocab)
    print(f'\n[Result] Test BLEU-1: {bleu1:.4f} | Test BLEU-4: {bleu4:.4f}')
    print(f'[Done] Best model saved to: {config.CHECKPOINT_PATH}')


train()