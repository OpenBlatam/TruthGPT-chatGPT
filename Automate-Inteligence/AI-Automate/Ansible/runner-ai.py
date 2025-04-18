import ansible.connection
import ansible.constants as C
import fnmatch
import multiprocessing
import os
import random
import jinja2
import time
import json
from ansible.utils import *
from ansible.errors import AnsibleInventoryNotFoundError


class Runner:
    ...  # Existing Ansible runner code here


class Pipeline(Runner):
    def __init__(self, *args, **kwargs):
        super(Pipeline, self).__init__(*args, **kwargs)
        self.steps = []

    def add_step(self, module_name, module_args):
        self.steps.append({ 'name': module_name, 'args': module_args})

    def run_pipeline(self, host_list):
        results = []
        self.host_list = self.parse_hosts(host_list)[0]
        for step in self.steps:
            self.module_name = step['name']
            self.module_args = step['args']
            results.append(self.run())
        return results


# Example of usage:
pipeline = Pipeline()
pipeline.add_step('command', 'uname -a')  # Add a step with 'command' module
pipeline.add_step('copy', 'src=/etc/hosts dest=/tmp/')  # Add a step with 'copy' module
results = pipeline.run_pipeline(['localhost', '127.0.0.1'])

for result in results:
    print(result['contacted'])