import logging
import pickle
from abc import ABC, abstractmethod
from queue import Queue, LifoQueue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ... Other command classes ...

class CommandManager:
    def __init__(self):
        self.command_queue = Queue()
        self.command_stack = LifoQueue()

    def add_command(self, command):
        self.save_command_state(command)
        self.command_queue.put(command)

    def execute_commands(self):
        while not self.command_queue.empty():
            command = self.command_queue.get()
            try:
                command.execute()
                self.command_stack.put(command)
            except Exception as e:
                logging.error(f'Error executing command {command.__class__.__name__}: {e}')

    def undo_last_command(self):
        if not self.command_stack.empty():
            command = self.command_stack.get()
            try:
                self.load_command_state(command)
                command.undo()
            except Exception as e:
                logging.error(f'Error undoing command {command.__class__.__name__}: {e}')

    def save_command_state(self, command):
        with open(f'{command.__class__.__name__}_state.pickle', 'wb') as f:
            pickle.dump(command, f)

    def load_command_state(self, command):
        with open(f'{command.__class__.__name__}_state.pickle', 'rb') as f:
            loaded_command = pickle.load(f)
            command.__dict__ = loaded_command.__dict__  # Update command state

# Using it:
command_manager = CommandManager()
logging_observer = LoggingObserver()

create_ticket_command = CommandFactory.create_command('create_ticket', {"system": "System1", "ticket_details": "details here"})
update_status_command = CommandFactory.create_command('update_node_status', {"system": "System1", "node_id": "123", "new_status": "active"})

create_ticket_command.subscribe(logging_observer)
update_status_command.subscribe(logging_observer)

command_manager.add_command(create_ticket_command)
command_manager.add_command(update_status_command)

command_manager.execute_commands()
command_manager.undo_last_command()