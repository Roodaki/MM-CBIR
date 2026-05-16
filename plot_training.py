import json
import matplotlib.pyplot as plt

with open("output/models/openai_clip-vit-base-patch32/training_log.json") as f:
    log = json.load(f)

epochs = [e["epoch"] for e in log]
train_loss = [e["train_loss"] for e in log]
val_loss = [e["val_loss"] for e in log]

plt.plot(epochs, train_loss, label="Train")
plt.plot(epochs, val_loss, label="Val")
plt.xlabel("Epoch")
plt.ylabel("InfoNCE Loss")
plt.legend()
plt.savefig("output/models/loss_curve.png", dpi=150)
