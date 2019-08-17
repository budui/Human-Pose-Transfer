from torchvision.utils import make_grid


def custom_global_step_transform(custom_period):
    """
    customize a global_step_transform for `ignite.contrib.handlers.BaseOutputHandler`,
    used to restore correct iteration or epoch when using CustomPeriodicEvent.
    :return: func:global_step_transform
    """

    def global_step_transform(engine, event_name):
        return engine.state.get_event_attrib_value(event_name) * custom_period

    return global_step_transform


def make_2d_grid(tensors, padding=0, normalize=True, range=None, scale_each=False, pad_value=0):
    # merge image in a batch in `y` direction first.
    grids = [make_grid(
        img_batch, padding=padding, nrow=1, normalize=normalize, range=range, scale_each=scale_each,
        pad_value=pad_value)
        for img_batch in tensors
    ]
    # merge images in `x` direction.
    return make_grid(grids, padding=0, nrow=len(grids))
