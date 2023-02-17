import os

import requests

from server.custom_logger import get_logger

backend_url = os.environ.get('BACKEND_URL')
logger = get_logger(__name__)


def login():
    credentials = {"email": os.environ.get('FLASK_MAIL'), "pw": os.environ.get('FLASK_PW')}
    login_url = f'{backend_url}/users/login'
    response = requests.post(login_url, json=credentials)

    if response.status_code == 204:
        return response.headers['session-token']
    else:
        logger.error('Could not login into the backend')
        raise ConnectionError('Could not login into the backend')


def set_run_status(run_name: str, status: str):
    try:
        token = login()
    except ConnectionError:
        return

    body = {'status': status}
    headers = {'Authorization': f'Bearer {token}'}
    endpoint_url = f'{backend_url}/runs/{run_name}/status'
    response = requests.put(endpoint_url, json=body, headers=headers)

    if response.status_code == 204:
        logger.info('Status of run "%s" set correctly to "%s" in the backend', run_name, status)
    else:
        logger.error('Could not set status of run "%s" in the backend', run_name)
