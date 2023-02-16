import multiprocessing
import os
from subprocess import Popen, PIPE

from server.custom_logger import get_logger


def _run_popnas_experiment(command: str, run_name: str):
    logger = get_logger('app')

    p = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = p.communicate()
    rcode = p.wait()

    if rcode == 0:
        logger.info('Run "%s" terminated successfully!', run_name)
    else:
        # use a different decoding on windows
        stderr = stderr.decode('cp437') if os.name == 'nt' else stderr.decode()

        logger.info('Run "%s" went wrong. Error code: %d', run_name, rcode)
        logger.error('Error: %s', stderr)


def launch_popnas_subprocess(command: str, run_name: str):
    proc = multiprocessing.Process(target=_run_popnas_experiment, args=(command, run_name))
    proc.start()

    return proc
