from optimization_core.factories.registry import Registry
from optimization_core.trainers.callbacks import PrintLogger, WandbLogger, TensorBoardLogger

CALLBACKS = Registry()


@CALLBACKS.register("print")
def build_print():
    return PrintLogger()


@CALLBACKS.register("wandb")
def build_wandb(project: str = None, run_name: str = None):
    return WandbLogger(project=project or "truthgpt", run_name=run_name or None)


@CALLBACKS.register("tensorboard")
def build_tensorboard(log_dir: str = None):
    return TensorBoardLogger(log_dir=log_dir or "runs")



