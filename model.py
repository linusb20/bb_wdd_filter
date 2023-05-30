import torch

class WDDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=5, padding="same"),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(8, 16, kernel_size=5, padding="same"),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.Conv3d(16, 32, kernel_size=5, padding="same"),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32 * 10 * 13 * 13, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # don't flatten batch dimension
        logits = self.classifier(x)
        return logits
