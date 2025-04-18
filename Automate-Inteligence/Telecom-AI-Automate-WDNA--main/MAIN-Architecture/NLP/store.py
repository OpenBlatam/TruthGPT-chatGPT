import os
import logging
import pickle
from abc import ABC, abstractmethod
from queue import Queue, LifoQueue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ... Other command and batch command classes ...

class CommandManager:
    def __init__(self):
        self.command_queue = Queue()
        self.done_stack = LifoQueue()
        self.undone_stack = LifoQueue()
        self.command_history = []

    # ... Other methods remain the same ...

    def store_command_history(self, filepath):
        with open(filepath, 'w') as file:
            for history in self.command_history:
                file.write(f'{history["action"]}: {history["command"]}\n')

# ... Using the CommandManager and CommandFactory with the commands code remains the same ...

# Storing command history
filepath = os.path.join(os.getenv('USERPROFILE'), 'Desktop', 'command_history.txt')  # Change the path as needed
command_manager.store_command_history(filepath)