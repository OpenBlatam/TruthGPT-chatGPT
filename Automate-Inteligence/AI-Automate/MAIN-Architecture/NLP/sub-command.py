class CommandManager:
    ...

    # Informal command string mapping to formal commands
    COMMAND_TRANSLATIONS = {'remove everything': 'delete', 'clean up': 'delete', 'look for': 'search', 'find': 'search'}

    ...

    def handle_text_command(self, text):
        # Lowercase the input for standard comparison
        text = text.lower()

        for informal, formal in self.COMMAND_TRANSLATIONS.items():
            if informal in text:
                if formal == 'delete':
                    # Call the actual delete command implementation here
                    self.execute_command(DeleteCommand('/path/to/delete'))

                elif formal == 'search':
                    # Call the actual search command implementation here
                    self.execute_command(SearchCommand('user query text'))

                return

        print("Sorry, I didn't understand that command.")