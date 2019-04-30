import os
import warnings


def make_handle_handle_exception(checkpoint_handler, save_networks, create_plots=None):
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            if create_plots is not None:
                create_plots(engine)

            exception_save_networks = {
                "exception_{}".format(k):save_networks[k]
                for k in save_networks
            }
            checkpoint_handler(engine, exception_save_networks)
        else:
            raise e
    return handle_exception


def make_handle_create_plots(output_dir, logs_path, plot_path):
    def create_plots(engine):
        try:
            import matplotlib as mpl
            mpl.use('agg')

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

        except ImportError:
            warnings.warn('Loss plots will not be generated -- pandas or matplotlib not found')

        else:
            df = pd.read_csv(os.path.join(output_dir, logs_path), delimiter='\t')
            # x = np.arange(1, engine.state.epoch * engine.state.iteration + 1, PRINT_FREQ)
            _ = df.plot(subplots=True, figsize=(10, 10))
            _ = plt.xlabel('Iteration number')
            fig = plt.gcf()
            path = os.path.join(output_dir, plot_path)

            fig.savefig(path)
            fig.clear()

    return create_plots


