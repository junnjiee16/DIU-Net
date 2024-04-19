import os
import json
from datetime import datetime


class Logger:
    def __init__(self):
        self.run_name = str(datetime.now().strftime("run_%d-%m-%Y_%H-%M"))

        # immediately create folder on intialization
        if not os.path.exists(f"./logs/{self.run_name}"):
            os.makedirs(f"./logs/{self.run_name}")

    def save_run(self, metadata):
        with open(f"./logs/{self.run_name}/results.json", "w") as outfile:
            json.dump(metadata, outfile)
