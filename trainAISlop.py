import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm  # <-- added


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("data/slop/train", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Training...")  # <-- was print(training)

model.train()
num_epochs = 5
for epoch in range(num_epochs):
    # progress bar for this epoch
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    running_loss = 0.0

    for images, labels in loop:
        images = images.to(device)
        # BCEWithLogitsLoss expects float targets of shape [batch_size, 1]
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # update the postfix shown at the end of the bar
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
