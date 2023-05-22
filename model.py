import torch

class WDDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(1,6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(6,16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool3d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 17 * 12 * 12, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # don't flatten batch dimension
        logits = self.classifier(x)
        return logits
