import multiprocessing
import os
from subprocess import Popen, PIPE, DEVNULL

from server.custom_logger import get_logger


def _run_popnas_experiment(command: str, run_name: str):
    logger = get_logger('app')

    p = Popen(command, stdout=DEVNULL, stderr=PIPE, shell=True)
    stdout, stderr = p.communicate()
    rcode = p.wait()

    if rcode == 0:
        logger.info('Run "%s" terminated successfully!', run_name)
    else:
        try:
            stderr = stderr.decode()
        except UnicodeDecodeError:
            logger.info('Could not decode correctly the stderr stream, using the byte representation')

        logger.info('Run "%s" went wrong. Error code: %d', run_name, rcode)
        logger.warning('Run "%s" error: %s', run_name, str(stderr))


def launch_popnas_subprocess(command: str, run_name: str):
    proc = multiprocessing.Process(target=_run_popnas_experiment, args=(command, run_name))
    proc.start()

    return proc
