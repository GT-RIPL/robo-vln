import torch


class _AuxLosses:
    def __init__(self):
        self._losses = {}
        self._loss_alphas = {}
        self._loss_masks = {}
        self._is_active = False

    def clear(self):
        self._losses.clear()
        self._loss_alphas.clear()

    def register_loss(self, name, loss, masks=None, alpha=1.0):
        assert self.is_active()
        assert name not in self._losses

        self._losses[name] = loss
        self._loss_alphas[name] = alpha
        # self._loss_masks[name] = masks


    def get_loss(self, name):
        return self._losses[name]

    def reduce(self, mask):
        assert self.is_active()
        total = 0.0
        for k in self._losses.keys():
            k_loss = torch.masked_select(self._losses[k], mask).mean()
            total = total + self._loss_alphas[k] * k_loss
        return total

    def reduce_parallel(self, mask):
        assert self.is_active()
        total = 0.0
        progress_total =[]
        for k in self._losses.keys():
            # k_loss = torch.masked_select(self._losses[k], self._loss_masks[k]).mean()
            # total = total + (self._loss_alphas[k] * k_loss).to('cuda:0')
            progress_total.append(self._losses[k].to('cuda:0'))
        progress_total = torch.cat(progress_total)
        k_loss = torch.masked_select(progress_total, mask).mean()
        total = total + (self._loss_alphas[k] * k_loss)
        return total

    def is_active(self):
        return self._is_active

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False


AuxLosses = _AuxLosses()
