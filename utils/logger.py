import os
import json
from datetime import datetime


class Logger:
    def __init__(self, metadata: dict):
        self.run_name = str(datetime.now().strftime("run_%d-%m-%Y_%H-%M"))
        self.metadata = metadata
        self.train_loss = []
        self.val_loss = []
        self.epochs_trained = 0
        self.best_epoch = 0

        # immediately create folder on intialization
        if not os.path.exists(f"./logs/{self.run_name}"):
            os.makedirs(f"./logs/{self.run_name}")

    def save_run(self):
        self.metadata["train_loss"] = self.train_loss
        self.metadata["val_loss"] = self.val_loss
        self.metadata["epochs_trained"] = self.epochs_trained

        with open(f"./logs/{self.run_name}/results.json", "w") as outfile:
            json.dump(self.metadata, outfile)
