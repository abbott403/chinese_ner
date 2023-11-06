import math
import numpy as np
from torch.optim.optimizer import Optimizer


class CustomDecayLR(object):
    """
    自定义学习率变化机制
        Example:
    #    >>> scheduler = CustomDecayLR(optimizer)
    #    >>> for epoch in range(100):
    #    >>>     scheduler.epoch_step()
    #    >>>     train(...)
    #    >>>         ...
    #    >>>         optimizer.zero_grad()
    #    >>>         loss.backward()
    #    >>>         optimizer.step()
    #    >>>     validate(...)
    """

    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def epoch_step(self, epoch):
        lr = self.lr
        if epoch > 12:
            lr = lr / 1000
        elif epoch > 8:
            lr = lr / 100
        elif epoch > 4:
            lr = lr / 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class BertLR(object):
    """
    Bert模型内定的学习率变化机制
    Example:
    #    >>> scheduler = BertLR(optimizer)
    #    >>> for epoch in range(100):
    #    >>>     scheduler.step()
    #    >>>     train(...)
    #    >>>         ...
    #    >>>         optimizer.zero_grad()
    #    >>>         loss.backward()
    #    >>>         optimizer.step()
    #    >>>         scheduler.batch_step()
    #    >>>     validate(...)
    """

    def __init__(self, optimizer, learning_rate, t_total, warmup):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.t_total = t_total
        self.warmup = warmup

    # 线性预热方式
    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def batch_step(self, training_step):
        lr_this_step = self.learning_rate * self.warmup_linear(training_step / self.t_total, self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step


class CyclicLR(object):
    """
    Cyclical learning rates for training neural networks
    Example:
    #    >>> scheduler = CyclicLR(optimizer)
    #    >>> for epoch in range(100):
    #    >>>     scheduler.step()
    #    >>>     train(...)
    #    >>>         ...
    #    >>>         optimizer.zero_grad()
    #    >>>         loss.backward()
    #    >>>         optimizer.step()
    #    >>>         scheduler.batch_step()
    #    >>>     validate(...)
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3, step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineLRWithRestarts(object):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will extend/shrink

    Example:
    #    >>> scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
    #    >>> for epoch in range(100):
    #    >>>     scheduler.step()
    #    >>>     train(...)
    #    >>>         ...
    #    >>>         optimizer.zero_grad()
    #    >>>         loss.backward()
    #    >>>         optimizer.step()
    #    >>>         scheduler.batch_step()
    #    >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, eta_threshold=1000, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.iteration = 0
        self.epoch_size = epoch_size
        self.eta_threshold = eta_threshold
        self.t_mult = t_mult
        self.verbose = verbose
        self.base_weight_decays = list(map(lambda group: group['weight_decay'], optimizer.param_groups))
        self.restart_period = restart_period
        self.restarts = 0
        self.t_epoch = -1
        self.batch_increments = []
        self._set_batch_increment()

    def _schedule_eta(self):
        """
        Threshold value could be adjusted to shrink eta_min and eta_max values.
        """
        eta_min = 0
        eta_max = 1
        if self.restarts <= self.eta_threshold:
            return eta_min, eta_max
        else:
            d = self.restarts - self.eta_threshold
            k = d * 0.09
            return eta_min + k, eta_max - k

    def get_lr(self, t_cur):
        eta_min, eta_max = self._schedule_eta()

        eta_t = (eta_min + 0.5 * (eta_max - eta_min)
                 * (1. + math.cos(math.pi *
                                  (t_cur / self.restart_period))))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        lrs = [base_lr * eta_t for base_lr in self.base_lrs]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart at epoch {}".format(self.last_epoch))
            self.restart_period *= self.t_mult
            self.restarts += 1
            self.t_epoch = 0

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = list(np.linspace(0, 1, batches_in_epoch))

    def batch_step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self.iteration += 1
        except IndexError:
            raise RuntimeError("Epoch size and batch size used in the "
                               "training loop and while initializing "
                               "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups, self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay


class NoamLR(object):
    """
    主要参考论文<< Attention Is All You Need>>中的学习更新方式
    Example:
   #     >>> scheduler = NoamLR(d_model,factor,warm_up,optimizer)
   #     >>> for epoch in range(100):
   #     >>>     scheduler.step()
   #     >>>     train(...)
   #     >>>         ...
   #     >>>         glopab_step += 1
   #     >>>         optimizer.zero_grad()
   #     >>>         loss.backward()
   #     >>>         optimizer.step()
   #     >>>         scheduler.batch_step(global_step)
   #     >>>     validate(...)
    """

    def __init__(self, d_model, factor, warm_up, optimizer):
        self.optimizer = optimizer
        self.warm_up = warm_up
        self.factor = factor
        self.d_model = d_model
        self._lr = 0

    def get_lr(self, step):
        lr = self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warm_up ** (-1.5)))
        return lr

    def batch_step(self, step):
        """
        update parameters and rate
        :return:
        """
        lr = self.get_lr(step)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr
