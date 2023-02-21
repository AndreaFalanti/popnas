from multiprocessing import Process
from subprocess import Popen, PIPE
from time import time

from flask import Flask, request
from flask_restful import Resource, Api, reqparse

import server.utils as U
from server.custom_logger import get_logger
from server.subprocesses import launch_popnas_subprocess
from server.tensorboard_p import TensorboardProcess, available_tb_ports, purge_inactive_tensorboard_processes, free_tensorboard_process

app = Flask(__name__)
api = Api(app)
logger = get_logger('app')

popnas_processes = {}   # type: dict[str, Process]
tensorboard_processes = {}  # type: dict[str, TensorboardProcess]

run_start_parser = reqparse.RequestParser()
run_start_parser.add_argument('name', type=str, required=True)
run_start_parser.add_argument('config_uri', type=str, required=True)


@app.after_request
def log_request(response):
    logger.info('Request (%s - %s) handled with status code: %d', request.method, request.path, response.status_code)
    return response


class ServerRoot(Resource):
    def get(self):
        return {'message': 'POPNAS Flask server is running'}


class Runs(Resource):
    def post(self):
        args = run_start_parser.parse_args(strict=True)
        run_name = args['name']

        command = U.build_popnas_run_command(run_name, args['config_uri'])
        popnas_processes[run_name] = launch_popnas_subprocess(command, run_name)
        logger.info('Started run %s execution', run_name)

        return {'name': run_name}, 201


class RunsTensorboard(Resource):
    def get(self, run_name: str):
        # TODO: avoid global
        global tensorboard_processes

        tb_proc = tensorboard_processes.get(run_name, None)
        if tb_proc:
            tb_proc.last_access = time()
            logger.info('Tensorboard instance already existing with port %d', tb_proc.port)
            return {'port': tb_proc.port}, 200
        else:
            tensorboard_processes = purge_inactive_tensorboard_processes(tensorboard_processes)

            port = available_tb_ports.pop()
            command = U.build_tensorboard_command(run_name, port)
            proc = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
            tensorboard_processes[run_name] = TensorboardProcess(proc, port, time())
            logger.info('Tensorboard instance created with port %d', port)

            return {'port': port}, 200

    def delete(self, run_name: str):
        # TODO: avoid global
        global tensorboard_processes

        tb_proc = tensorboard_processes.get(run_name, None)
        if tb_proc:
            free_tensorboard_process(tb_proc, run_name)
            return {}, 204
        else:
            return {'error': f'No tensorboard instance of run "{run_name}" is currently executing'}, 404


class RunsStop(Resource):
    def post(self, run_name: str):
        popnas_proc = popnas_processes.get(run_name, None)
        if popnas_proc:
            popnas_proc.kill()
            del popnas_processes[run_name]
            logger.info('POPNAS run %s stopped successfully', run_name)

            return {}, 204
        else:
            logger.info('No run with given name: %s', run_name)
            return {'error': f'No run with given name: {run_name}'}, 400


class RunsResume(Resource):
    def post(self, run_name: str):
        popnas_proc = popnas_processes.get(run_name, None)
        if popnas_proc:
            logger.info('Run %s is already in progress!', run_name)
            return {'error': f'Run {run_name} is already in progress!'}, 400
        else:
            command = U.build_popnas_restore_command(run_name)
            popnas_processes[run_name] = launch_popnas_subprocess(command, run_name)
            logger.info('Run %s resumed successfully', run_name)
            return {}, 204


api.add_resource(ServerRoot, '/')
api.add_resource(Runs, '/runs')
api.add_resource(RunsTensorboard, '/runs/<string:run_name>/tensorboard')
api.add_resource(RunsStop, '/runs/<string:run_name>/stop')
api.add_resource(RunsResume, '/runs/<string:run_name>/resume')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
