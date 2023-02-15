import shlex
from subprocess import Popen, PIPE
from time import time

from flask import Flask
from flask_restful import Resource, Api, reqparse

from server.tensorboard_p import TensorboardProcess, available_tb_ports, purge_inactive_tensorboard_processes

app = Flask(__name__)
api = Api(app)

popnas_processes = {}   # type: 'dict[str, Popen]'
tensorboard_processes = {}  # type: 'dict[str, TensorboardProcess]'

run_start_parser = reqparse.RequestParser()
run_start_parser.add_argument('name', type=str, required=True)
run_start_parser.add_argument('config_uri', type=str, required=True)


def build_popnas_run_command(run_name: str, config_uri: str):
    return shlex.split(f'python run.py -j {config_uri} -name {run_name}')


def build_popnas_restore_command(run_name: str):
    return shlex.split(f'python run.py -r logs/{run_name}')


def build_tensorboard_command(run_name: str, port: int):
    return shlex.split(f'tensorboard --logdir logs/{run_name}/tensorboard_cnn --port {port} --bind_all')


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class Runs(Resource):
    def post(self):
        args = run_start_parser.parse_args(strict=True)

        command = build_popnas_run_command(args['name'], args['config_uri'])
        # popnas_processes[args['name']] = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        #
        # return {'name': args['name']}, 201
        return {'command': command}, 201


class RunsTensorboard(Resource):
    def get(self, run_name: str):
        # TODO: avoid global
        global tensorboard_processes

        tb_proc = tensorboard_processes.get(run_name, None)
        if tb_proc:
            tb_proc.last_access = time()
            return {'port': tb_proc.port}, 200
        else:
            tensorboard_processes = purge_inactive_tensorboard_processes(tensorboard_processes)

            port = available_tb_ports.pop()
            command = build_tensorboard_command(run_name, port)
            proc = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
            tensorboard_processes[run_name] = TensorboardProcess(proc, port, time())

            return {'port': port}, 200


class RunsStop(Resource):
    def post(self, run_name: str):
        popnas_proc = popnas_processes.get(run_name, None)
        if popnas_proc:
            popnas_proc.kill()
            del popnas_processes[run_name]
            return {}, 204
        else:
            print(f'No run with given name: {run_name}')
            return {'error': f'No run with given name: {run_name}'}, 400


class RunsResume(Resource):
    def post(self, run_name: str):
        popnas_proc = popnas_processes.get(run_name, None)
        if popnas_proc:
            return {'error': f'Run {run_name} is already in progress!'}, 400
        else:
            command = build_popnas_restore_command(run_name)
            popnas_processes[run_name] = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
            return {}, 204


api.add_resource(HelloWorld, '/')
api.add_resource(Runs, '/runs')
api.add_resource(RunsTensorboard, '/runs/<string:run_name>/tensorboard')
api.add_resource(RunsStop, '/runs/<string:run_name>/stop')
api.add_resource(RunsResume, '/runs/<string:run_name>/resume')
