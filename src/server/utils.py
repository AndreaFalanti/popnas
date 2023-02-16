def build_popnas_run_command(run_name: str, config_uri: str):
    return f'python run.py -j {config_uri} --name {run_name}'


def build_popnas_restore_command(run_name: str):
    return f'python run.py -r logs/{run_name}'


def build_tensorboard_command(run_name: str, port: int):
    return f'tensorboard --logdir logs/{run_name}/tensorboard_cnn --port {port} --bind_all'
