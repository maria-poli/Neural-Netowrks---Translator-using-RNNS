import matplotlib.pyplot as plt

data = """
Epoch: 01, Train Loss: 3.721, Val Loss: 2.727, BLEU: 0.507, Perplexity: 15.529
Epoch: 02, Train Loss: 2.572, Val Loss: 2.553, BLEU: 0.520, Perplexity: 13.042
Epoch: 03, Train Loss: 2.458, Val Loss: 2.477, BLEU: 0.530, Perplexity: 12.077
Epoch: 04, Train Loss: 2.392, Val Loss: 2.428, BLEU: 0.536, Perplexity: 11.497
Epoch: 05, Train Loss: 2.343, Val Loss: 2.391, BLEU: 0.546, Perplexity: 11.069
Epoch: 06, Train Loss: 2.303, Val Loss: 2.367, BLEU: 0.554, Perplexity: 10.806
Epoch: 07, Train Loss: 2.269, Val Loss: 2.335, BLEU: 0.555, Perplexity: 10.462
Epoch: 08, Train Loss: 2.233, Val Loss: 2.312, BLEU: 0.567, Perplexity: 10.216
Epoch: 09, Train Loss: 2.199, Val Loss: 2.290, BLEU: 0.573, Perplexity: 9.994
Epoch: 10, Train Loss: 2.168, Val Loss: 2.275, BLEU: 0.579, Perplexity: 9.839
"""

model = "bidirectional_2"

# Parse the data
epochs = []
train_losses = []
val_losses = []
bleu_scores = []
perplexities = []

for line in data.strip().split("\n"):
    parts = line.split(", ")
    epoch = int(parts[0].split(": ")[1])
    train_loss = float(parts[1].split(": ")[1])
    val_loss = float(parts[2].split(": ")[1])
    bleu = float(parts[3].split(": ")[1])
    perplexity = float(parts[4].split(": ")[1])

    epochs.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    bleu_scores.append(bleu)
    perplexities.append(perplexity)

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig(f'./results/loss_curves_{model}.png')
plt.close()
