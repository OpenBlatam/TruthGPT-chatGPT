import ansible_runner
import json


import multiprocessing
import random
import time

def run_module(module, args):
    r = ansible_runner.run(private_data_dir='/tmp', host_pattern='localhost', module=module, module_args=args)
    return json.dumps(dict(r.items()))


def main():
    run_module('command', 'uname -a')
    run_module('copy', 'src=/etc/hosts, dest=/tmp')


class Runner:
    ...  # Existing Ansible runner code here


class Pipeline(Runner):
    def __init__(self, *args, **kwargs):
        super(Pipeline, self).__init__(*args, **kwargs)
        self.steps = []

    def add_step(self, module_name, module_args):
        self.steps.append({'name': module_name, 'args': module_args})

    def run_pipeline(self, host_list):
        results = []
        self.host_list = self.parse_hosts(host_list)[0]
        for step in self.steps:
            self.module_name = step['name']
            self.module_args = step['args']
            results.append(self.run())
        return results


# Use regular dict instead of ImmutableDict
context.CLIARGS = dict(connection='local', module_path=['/to/mymodules'], forks=10, become=None,
                       become_method=None, become_user=None, check=False, diff=False)

pipeline = Pipeline()
pipeline.add_step('command', 'uname -a')  # Add a step with 'command' module
pipeline.add_step('copy', 'src=/etc/hosts dest=/tmp/')  # Add a step with 'copy' module

# Note here we pass in list of hosts.
results = pipeline.run_pipeline(['localhost', '127.0.0.1'])

for result in results:
    print(result['contacted'])