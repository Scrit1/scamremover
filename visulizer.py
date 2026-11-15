import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
import torchvision.models as tvm

# 1. Build the architecture to MATCH your checkpoint (1 output unit)
model = tvm.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)  # <-- Changed to 1 to match checkpoint

# 2. Load the checkpoint safely
sd = torch.load('/Users/displayaccount/Downloads/resnet18_cifake.pth', map_location='cpu')
sd = sd.get('state_dict', sd) # Unwrap if saved inside a dict
sd = {k.replace('module.', ''): v for k, v in sd.items()} # Strip DataParallel wrapper

model.load_state_dict(sd)
model.eval()

# 3. Extract weights from the dense head (fc.weight is shape [1, 512])
W_fc = model.fc.weight.detach().numpy()  # Shape: (1, 512)
b_fc = model.fc.bias.detach().numpy()    # Shape: (1,)

# Since there is only 1 output, we take the first row for our weights
w = W_fc[0] 

# 4. Pick the top 25 most influential features (out of the 512)
N_FEAT = 25
feat_idx = np.argsort(np.abs(w))[-N_FEAT:]

# 5. Setup the figure
fig, ax = plt.subplots(figsize=(13, 8))
xs = [0, 1]; sizes = [N_FEAT, 2]
pos = {}
for li, (x, n) in enumerate(zip(xs, sizes)):
    for j, y in enumerate(np.linspace(0.05, 0.95, n)):
        pos[(li, j)] = (x, y)

wmax = np.abs(w).max()

# 6. Draw edges from features to the two visual output nodes
for i, fi in enumerate(feat_idx):
    val = w[fi]
    if abs(val) < 0.10 * wmax:  # Ignore very weak connections to keep the plot clean
        continue
        
    # Top node ("fake"): positive weights mean pushing towards fake
    ax.plot(*zip(pos[(0, i)], pos[(1, 0)]),
            color='tab:red' if val > 0 else 'tab:blue',
            lw=abs(val) / wmax * 5, alpha=.5, zorder=1)
            
    # Bottom node ("real"): we invert the color visually because 
    # positive weights push away from real (ham)
    ax.plot(*zip(pos[(0, i)], pos[(1, 1)]),
            color='tab:blue' if val > 0 else 'tab:red',
            lw=abs(val) / wmax * 5, alpha=.5, zorder=1)

# 7. Draw Feature Nodes
for i, fi in enumerate(feat_idx):
    x, y = pos[(0, i)]
    ax.scatter(x, y, s=200, c='lightblue', edgecolors='k', zorder=3)
    ax.text(x - 0.04, y, f'feat{fi}', ha='right', va='center', fontsize=8)

# 8. Draw Output Nodes
ax.scatter(*pos[(1, 0)], s=500, c='gold', edgecolors='k', zorder=3)
ax.text(1.05, pos[(1, 0)][1], f'fake (b={b_fc[0]:+.2f})', va='center', fontsize=10, fontweight='bold')

ax.scatter(*pos[(1, 1)], s=500, c='lightgray', edgecolors='k', zorder=3)
ax.text(1.05, pos[(1, 1)][1], f'real (1-p)', va='center', fontsize=10, fontweight='bold')

# 9. Final formatting
ax.set_xlim(-0.3, 1.5); ax.set_ylim(0, 1); ax.axis('off')
ax.set_title('ResNet18 CIFAKE classifier head (red=+ weight, blue=- weight, thickness=|weight|)')
plt.tight_layout()
plt.savefig('wiring.png', dpi=150)
plt.show()
