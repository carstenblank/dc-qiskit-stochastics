import logging

import qiskit
from qiskit.providers import ibmq

LOG = logging.getLogger(__name__)

_ibmq_config = {
    'hub': None,
    'group': None,
    'project': None
}

if qiskit.IBMQ.active_account() is None or len(qiskit.IBMQ.active_account()) == 0:
    LOG.info('No active IBMQ account found, activating...')
    if len(qiskit.IBMQ.stored_account()) > 0:
        qiskit.IBMQ.load_account()
    else:
        LOG.warning('No qiskit accounts found!')


def set_provider_config(hub=None, group=None, project=None):
    global _ibmq_config
    _ibmq_config['hub'] = hub
    _ibmq_config['group'] = group
    _ibmq_config['project'] = project


def provider() -> 'ibmq.AccountProvider':
    LOG.debug(f'Requesting IBMQ provider.')
    active_account = qiskit.IBMQ.active_account()
    if active_account is None:
        raise ValueError("No active accounts there!")
    LOG.debug(f'Active IBMQ account found: {active_account}')
    global _ibmq_config
    return qiskit.IBMQ.get_provider(hub=_ibmq_config.get('hub'),
                                    group=_ibmq_config.get('group'),
                                    project=_ibmq_config.get('project'))
