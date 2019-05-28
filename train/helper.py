import torch
from ignite.metrics import RunningAverage

def move_data_pair_to(device, data_pair):
    # move data to GPU
    for k in data_pair:
        if "path" in k:
            # do not move string
            continue
        elif isinstance(data_pair[k], dict):
            for kk, v in data_pair[k].items():
                data_pair[k][kk] = v.to(device)
        else:
            data_pair[k] = data_pair[k].to(device)


class LossContainer(object):
    def __init__(self, loss, loss_lambda, print_self=True, device="cuda"):
        self.loss = loss
        self.loss.to(device)
        self._lambda = loss_lambda
        self.activated = self._lambda > 0
        self.fake_loss = torch.zeros([1], device=device, requires_grad=False, dtype=torch.float)

        if print_self and self.activated:
            print("Loss: {} * {}".format(type(self.loss).__name__, self._lambda))
            print(self.loss)

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs) * self._lambda if self.activated else self.fake_loss

    def __repr__(self):
        return "LossContainer <{}> * {}".format(type(self.loss).__name__, self._lambda)


def attach_engine(engine, ra):
    for name, fn in ra.items():
        print("attach RunningAverage: {}".format(name))
        RunningAverage(output_transform=fn).attach(engine, name)