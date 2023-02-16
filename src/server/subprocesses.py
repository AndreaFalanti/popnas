import multiprocessing
import os
import sys
from subprocess import Popen, PIPE


def _run_popnas_experiment(command: str, run_name: str):
    p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = p.communicate()
    rcode = p.wait()

    if rcode == 0:
        print(f'Run "{run_name}" terminated successfully!')
    else:
        # use a different decoding on windows
        stderr = stderr.decode('cp437') if os.name == 'nt' else stderr.decode()

        print(f'Run "{run_name}" went wrong. Error code: {rcode}')
        print(f'Error: {stderr}', file=sys.stderr)


def launch_popnas_subprocess(command: str, run_name: str):
    proc = multiprocessing.Process(target=_run_popnas_experiment, args=(command, run_name))
    proc.start()

    return proc
