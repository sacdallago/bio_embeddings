import torch.nn as nn


class ConservationCNN(nn.Module):
    """Convolutional neural network for prediction of conservation scores from 0-8 (0=variable; 8=conserved)"""
    n_features = 1024
    bottleneck_dim = 32
    n_classes = 9
    dropout_rate = 0.25

    def __init__(self):
        super(ConservationCNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.n_features, self.bottleneck_dim, kernel_size=(7, 1), padding=(3, 0)
            ),  # 7x32
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(
                self.bottleneck_dim, self.n_classes, kernel_size=(7, 1), padding=(3, 0)
            ),
        )

    def forward(self, x):
        """
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        """
        # IN: X = (L x F); OUT: (1 x F x L, 1)
        x = x.unsqueeze(dim=0).permute(0, 2, 1).unsqueeze(dim=-1)
        Yhat_consurf = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)
        # IN: (B x N x L x 1); OUT: ( B x L x N )
        Yhat_consurf = Yhat_consurf.squeeze(dim=-1)
        return Yhat_consurf
