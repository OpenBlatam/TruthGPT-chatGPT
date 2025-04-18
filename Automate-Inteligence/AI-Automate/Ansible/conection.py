class Connection(object):
    ''' Handles abstract connections to remote hosts '''

# Create the different ways for conect to petitions
    def connect(self, host):
        conn = None
        if self.transport == 'paramiko':
            conn = ParamikoConnection(self.runner, host)
        # Add your own connection types here
        elif self.transport == 'mat':
            conn = MatConnection(self.runner, host)
        elif self.transport == 'remedy':
            conn = RemedyConnection(self.runner, host)
        elif self.transport == 'gestor_mae':
            conn = GestorMaeConnection(self.runner, host)
        # -----------------------------
        if conn is None:
            raise Exception("unsupported connection type")
        return conn.connect()