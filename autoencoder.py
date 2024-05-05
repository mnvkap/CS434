import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class Autoencoder(nn.Module):
    def __init__(self, input_size=80, hidden_size=40):
        super(Autoencoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        encode_layers = [
            nn.Linear(input_size, 60),
            nn.ReLU(),
            nn.Linear(60, hidden_size),
            nn.ReLU(),
        ]
        decode_layers = [
            nn.Linear(hidden_size, 60),
            nn.ReLU(),
            nn.Linear(60, input_size),
        ]

        self.encoder = nn.Sequential(*encode_layers).to(self.device)
        self.decoder = nn.Sequential(*decode_layers).to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def _evaluate_dl(self, dataloader, criterion):
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                running_loss += loss.item()

        return running_loss

    def fit(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs=100,
        lr=0.001,
        stat_interval=500,
    ):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        print("Training...")
        for epoch in range(epochs):

            running_loss = 0.0
            for i, data in enumerate(train_dataloader):
                inputs, _ = data  # We don't need the labels for reconstruction loss
                inputs = inputs.to(self.device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % stat_interval == 0:
                    print(
                        f"[E:{epoch+1}\tB:{i + 1}], Loss: {running_loss / stat_interval:.3f}"
                    )
                    running_loss = 0.0

            if val_dataloader:
                print("Evaluating on validation set...")
                val_loss = self._evaluate_dl(val_dataloader, criterion)
                print(f"Validation Loss: {val_loss:.3f}")


class Classifier(nn.Module):
    """This class is used to train a classifier with pretrained autoencoder embeddings."""

    def __init__(self, device: str | None = None):
        super(Classifier, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if device:
            self.device = torch.device(device)

        num_labels = 17
        num_inputs = 80

        self.train_loss_values = []
        self.val_loss_values = []

        self.model = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

    def confusion_matrix(self, dataloader):
        """Generate a confusion matrix for the dataloader."""
        conf_matrix = torch.zeros(
            17, 17, dtype=torch.int64
        )  # Assuming num_labels is 17
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    conf_matrix[t.long(), p.long()] += 1
        return conf_matrix

    def _evaluate_dl(self, dataloader, criterion):
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss

    def eval_dl(self, dataloader):
        """Evaluate the model on a dataloader and return the accuracy."""
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def fit(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs=100,
        lr=0.001,
        stat_interval=500,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.parameters(), lr=lr
        )  # This is also training the autoencoder

        self.train_loss_values = []
        self.val_loss_values = []

        for epoch in range(epochs):
            running_loss = 0.0
            self.train()
            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % stat_interval == 0:
                    print(
                        f"[E: {epoch+1}\tB:{i + 1:5d}], Loss: {running_loss / stat_interval:.3f}"
                    )
                    self.train_loss_values.append(running_loss / stat_interval)
                    running_loss = 0.0

            if val_dataloader:
                self.eval()
                print("Evaluating on validation set...")
                val_loss = self._evaluate_dl(val_dataloader, criterion)
                print(f"Validation Loss: {val_loss / len(val_dataloader):.3f}")


class ConvClassifier(nn.Module):
    """This class is used to train a classifier with convolutions on the raw data."""

    def __init__(self, device: str | None = None):
        super(ConvClassifier, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if device:
            self.device = torch.device(device)

        num_labels = 17
        num_inputs = 80

        self.train_loss_values = []
        self.val_loss_values = []

        # self.model = nn.Sequential(
        #     nn.Linear(num_inputs, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, num_labels),
        #     # nn.Softmax(dim=1),
        # ).to(self.device)

        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # Output: [batch_size, 32, 80]
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output: [batch_size, 32, 40]
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # Output: [batch_size, 64, 40]
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output: [batch_size, 64, 20]
            nn.Conv1d(
                64, 128, kernel_size=3, padding=1
            ),  # Output: [batch_size, 128, 20]
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output: [batch_size, 128, 10]
            nn.Flatten(),  # Flatten the output for the dense layer
            nn.Linear(1280, 256),  # 128 channels * 10 features
            nn.ReLU(),
            nn.Linear(256, num_labels),
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

    def _evaluate_dl(self, dataloader, criterion):
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss

    def eval_dl(self, dataloader):
        """Evaluate the model on a dataloader and return the accuracy."""
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def fit(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs=100,
        lr=0.001,
        stat_interval=500,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.parameters(), lr=lr
        )  # This is also training the autoencoder

        self.train_loss_values = []
        self.val_loss_values = []

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader):
                inputs, labels = data

                inputs = inputs.view(inputs.shape[0], 1, -1)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % stat_interval == 0:
                    print(
                        f"[E: {epoch+1}\tB:{i + 1:5d}], Loss: {running_loss / stat_interval:.3f}"
                    )
                    self.train_loss_values.append(running_loss / stat_interval)
                    running_loss = 0.0

            if val_dataloader:
                print("Evaluating on validation set...")
                val_loss = self._evaluate_dl(val_dataloader, criterion)
                print(f"Validation Loss: {val_loss / len(val_dataloader):.3f}")


if __name__ == "__main__":
    # Use torchviz to visualize the model 'Classifier'
    from torchviz import make_dot

    model = Classifier()
    x = torch.randn(1, 80).to(model.device)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters())).render(
        "DNN Classifier", format="png"
    )
