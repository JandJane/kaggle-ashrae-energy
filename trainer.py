# Abstract class for all model types
class Trainer:
    def __init__(self):
        pass

    def train(self, verbose=True, do_val=True, *args, **kwargs):
        pass

    def plot(self, pic_name):
        pass

    def predict(self, test_df, submission, batch_size=100000):
        pass

    def save_model(self, name):
        pass

    def load_model(self, name):
        pass
