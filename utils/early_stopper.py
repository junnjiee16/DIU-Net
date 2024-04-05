class EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def early_stop(self, curr_val_loss: float):
        if curr_val_loss < self.best_loss:
            self.best_loss = curr_val_loss
            self.counter = 0
        elif curr_val_loss > (self.best_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
