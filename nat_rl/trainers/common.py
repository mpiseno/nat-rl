


class BaseTrainer:
    def train(self):
        raise NotImplementedError

    def eval(self):
        print(f'evaluating!')