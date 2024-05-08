import os
import json
from datetime import datetime


class Logger:
    def __init__(self, logdir=None):
        if logdir == None:
            self.logdir = f"./logs/{str(datetime.now().strftime('run_%d-%m-%Y_%H-%M'))}"
            # immediately create folder on intialization
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
        else:
            self.logdir = logdir

    def save_run(self, metadata):
        with open(f"{self.logdir}/results.json", "w") as outfile:
            json.dump(metadata, outfile)
