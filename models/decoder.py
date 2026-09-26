import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_size, attention_dim=256):
        super(Attention, self).__init__()
        # Project encoder features (49 regions) into attention space
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # Project LSTM hidden state into attention space
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        # Produce a single score per region
        self.full_att    = nn.Linear(attention_dim, 1)
        self.relu        = nn.ReLU()

    def forward(self, encoder_out, hidden):
        
        # Score each of the 49 regions
        att1 = self.encoder_att(encoder_out)              # (batch, 49, attention_dim)
        att2 = self.decoder_att(hidden).unsqueeze(1)      # (batch, 1,  attention_dim)
        scores = self.full_att(self.relu(att1 + att2))    # (batch, 49, 1)
        alpha  = F.softmax(scores.squeeze(2), dim=1)      # (batch, 49)

        # Weighted sum of image regions
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        return context, alpha

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,
                 num_layers=config.NUM_LAYERS, dropout=config.DROPOUT):
        super(DecoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.embed_size  = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        self.attention = Attention(
            encoder_dim  = embed_size,
            hidden_size  = hidden_size,
            attention_dim= 256,
        )

        # LSTM takes [word_embed + context_vector] as input
        self.lstm_cell = nn.LSTMCell(
            input_size  = embed_size + embed_size,
            hidden_size = hidden_size,
        )

        # Initialize hidden and cell state from mean of encoder features
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

        # Project LSTM output → vocab scores
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def _init_hidden(self, encoder_out):
        """Initialize LSTM states from mean of image features."""
        mean_enc = encoder_out.mean(dim=1)              # (batch, embed_size)
        h = torch.tanh(self.init_h(mean_enc))           # (batch, hidden_size)
        c = torch.tanh(self.init_c(mean_enc))           # (batch, hidden_size)
        return h, c

    def forward(self, encoder_out, captions):
       
        batch_size = encoder_out.size(0)
        seq_len    = captions.size(1) - 1      # exclude <EOS> from inputs

        # Embed all caption words at once
        embeds = self.dropout(self.embedding(captions[:, :-1]))  # (batch, seq_len, embed_size)

        h, c = self._init_hidden(encoder_out)

        logits = torch.zeros(batch_size, seq_len, self.vocab_size).to(encoder_out.device)

        for t in range(seq_len):
            # Attention: which image regions matter for this word?
            context, _ = self.attention(encoder_out, h)   # (batch, embed_size)

            # Concatenate word embedding + context vector
            lstm_input = torch.cat([embeds[:, t, :], context], dim=1)  # (batch, embed_size*2)

            h, c = self.lstm_cell(lstm_input, (h, c))     # (batch, hidden_size)

            logits[:, t, :] = self.fc(self.dropout(h))    # (batch, vocab_size)

        return logits

    def generate(self, encoder_out, vocab, max_len=config.MAX_CAPTION_LEN):
   
        self.eval()
        generated  = []
        seen_words = {}    # track word frequencies for repetition penalty

        with torch.no_grad():
            h, c = self._init_hidden(encoder_out)

        # Start with <SOS> token
            word_idx = torch.tensor([vocab.word2idx["<SOS>"]], device=encoder_out.device)

        for _ in range(max_len):
            embed          = self.dropout(self.embedding(word_idx))
            context, alpha = self.attention(encoder_out, h)
            lstm_input     = torch.cat([embed, context], dim=1)
            h, c           = self.lstm_cell(lstm_input, (h, c))
            logits         = self.fc(h)                          # (1, vocab_size)

            # ── Block <UNK> completely ─────────────────────────────────────
            logits[0, vocab.word2idx["<UNK>"]] = float('-inf')

            # ── Repetition penalty ─────────────────────────────────────────
            for prev_idx, count in seen_words.items():
                logits[0, prev_idx] /= (1.5 * count)

            # ── Block "a" repeating after itself ───────────────────────────
            if generated and generated[-1] == vocab.word2idx.get("a", -1):
                logits[0, vocab.word2idx.get("a", 0)] = float('-inf')

            word_idx  = logits.argmax(dim=1)
            predicted = word_idx.item()

            if predicted == vocab.word2idx["<EOS>"]:
                break

            generated.append(predicted)
            seen_words[predicted] = seen_words.get(predicted, 0) + 1

        return vocab.decode(generated)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        logits   = self.decoder(features, captions)
        return logits

    def generate_caption(self, image, vocab):
        features = self.encoder(image)
        return self.decoder.generate(features, vocab)