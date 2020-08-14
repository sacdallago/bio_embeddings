import torch.nn as nn


class SUBCELL_FNN(nn.Module):
    # in 10 states and a binary classification into membrane-bound vs. soluble
    def __init__(self, use_batch_norm=True):
        super(SUBCELL_FNN, self).__init__()
        # Linear layer, taking embedding dimension 1024 to make predictions:
        if use_batch_norm:
            self.layer = nn.Sequential(
                nn.Linear(1024, 32),  # in, out
                nn.Dropout(0.25),  # dropout
                nn.ReLU(),
                nn.BatchNorm1d(32)
                )
        else:
            self.layer = nn.Sequential(
                nn.Linear(1024, 32),  # in, out
                nn.Dropout(0.25),  # dropout
                nn.ReLU(),
                )
        self.loc_classifier = nn.Linear(32, 10)
        self.mem_classifier = nn.Linear(32, 2)

    def forward(self, x):
        out = self.layer(x)  # map 1024-dimensional ELMo vector to 32-dims
        # based on 32 dims, predict localization and membrane-bound
        Yhat_loc = self.loc_classifier(out)
        Yhat_mem = self.mem_classifier(out)

        return Yhat_loc, Yhat_mem


class SECSTRUCT_CNN(nn.Module):
    # Convolutional neural network for prediction of Sec.Struct. in 3- & 8-states and disorder
    def __init__(self):
        super(SECSTRUCT_CNN, self).__init__()

        self.elmo_feature_extractor = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.dssp3_classifier = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(7, 1), padding=(3, 0))
        )
        self.dssp8_classifier = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        x = self.elmo_feature_extractor(x)  # compress ELMo features to 32-dims

        d3_Yhat = self.dssp3_classifier(x)
        d8_Yhat = self.dssp8_classifier(x)
        diso_Yhat = self.diso_classifier(x)

        return d3_Yhat, d8_Yhat, diso_Yhat
