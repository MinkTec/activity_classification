import math


def lrfn(current_step, num_warmup_steps, lr_max, epochs, num_cycles=0.50):
    """
    learning rate scheduler
    https://www.kaggle.com/code/akshatpattiwar/hubmap-tensorflow/notebook
    """

    if current_step < num_warmup_steps:
        return lr_max * 0.50 ** (num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, epochs - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
