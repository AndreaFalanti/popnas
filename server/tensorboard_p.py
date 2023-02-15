from dataclasses import dataclass
from subprocess import Popen
from time import time

# ports that can be used to expose Tensorboard dashboards
available_tb_ports = list(range(7090, 7095))
threshold_time = 60 * 15    # 15 minutes


@dataclass
class TensorboardProcess:
    proc: Popen
    port: int
    last_access: float


def free_tensorboard_process(tb_process: TensorboardProcess, run_name: str):
    available_tb_ports.append(tb_process.port)
    tb_process.proc.kill()
    print(f'Tensorboard instance of run {run_name} stopped')


def purge_inactive_tensorboard_processes(tb_processes: 'dict[str, TensorboardProcess]'):
    new_processes = {}
    for run_name, v in tb_processes.items():
        if time() - v.last_access >= threshold_time:
            free_tensorboard_process(v, run_name)
        else:
            new_processes[run_name] = v

    return new_processes
