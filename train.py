import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset import CharDataset
from src.model.transformer_model import TransformerLanguageModel
from src.model.attention import causal_mask


# ======================================================
# CONFIG
# ======================================================
SEQ_LEN = 128
BATCH_SIZE = 32
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 256
EPOCHS = 5
LR = 3e-4
GRAD_CLIP = 1.0

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"   # AMP only safe on CUDA

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ======================================================
# DATA
# ======================================================
with open("data/wikitext-2-raw/wiki.train.raw", "r", encoding="utf-8") as f:
    train_text = f.read()

with open("data/wikitext-2-raw/wiki.valid.raw", "r", encoding="utf-8") as f:
    val_text = f.read()

# ---- optional debug mode (recommended initially) ----
# train_text = train_text[:500_000]
# val_text   = val_text[:100_000]

train_dataset = CharDataset(train_text, SEQ_LEN)
val_dataset   = CharDataset(val_text, SEQ_LEN)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
)

VOCAB_SIZE = train_dataset.vocab_size


# ======================================================
# MODEL
# ======================================================
model = TransformerLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_hidden_dim=FF_DIM,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
    verbose=True,
)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


# ======================================================
# CHECKPOINTING
# ======================================================
def save_checkpoint(epoch, model, optimizer, train_loss, val_loss):
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


# ======================================================
# VALIDATION
# ======================================================
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        mask = causal_mask(x.size(1), DEVICE)
        logits = model(x, mask)

        loss = criterion(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1)
        )

        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)


# ======================================================
# TEXT GENERATION
# ======================================================
@torch.no_grad()
def generate_text(
    model,
    dataset,
    start_text="The ",
    length=300,
    temperature=1.0,
):
    model.eval()

    indices = [dataset.stoi[c] for c in start_text]
    x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(length):
        mask = causal_mask(x.size(1), DEVICE)
        logits = model(x, mask)

        next_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_idx = torch.multinomial(probs, 1).item()

        x = torch.cat(
            [x, torch.tensor([[next_idx]], device=DEVICE)],
            dim=1
        )

    text = "".join(dataset.itos[i] for i in x[0].tolist())
    print("\n===== GENERATED TEXT =====\n")
    print(text)
    print("\n==========================\n")


# ======================================================
# TRAINING LOOP
# ======================================================
train_losses = []
val_losses = []

model.train()
print("Number of batches per epoch:", len(train_loader))

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        mask = causal_mask(x.size(1), DEVICE)
        optimizer.zero_grad()

        if USE_AMP:
            with torch.cuda.amp.autocast():
                logits = model(x, mask)
                loss = criterion(
                    logits.view(-1, VOCAB_SIZE),
                    y.view(-1)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, mask)
            loss = criterion(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % 1000 == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx} | "
                f"Loss {loss.item():.4f}"
            )

    train_loss = total_loss / len(train_loader)
    val_loss = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"\nEpoch {epoch} DONE | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}\n"
    )

    scheduler.step(val_loss)
    save_checkpoint(epoch, model, optimizer, train_loss, val_loss)

    generate_text(model, train_dataset, start_text="The ", length=200)


# ======================================================
# PLOT TRAINING + VALIDATION LOSS
# ======================================================
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_curve.png", dpi=300)
plt.show()
