class Cluster:

    def __init__(self, lambd, num_clusters):
        self.mu_tilde = np.zeros(num_clusters)
        self.c_mu = np.zeros(num_clusters)
        self.lambd = lambd

    def update(self, queries_keys):
        """
        Assume queries_keys is a 2D numpy array,
        where rows represent different i:s,
        and the first column represents 𝐪𝑖 and
        and the second column represents 𝐤𝑗.
        """
        for i in range(queries_keys.shape[0]):
            mu_qi = self.get_mu(queries_keys[i, 0])
            mu_kj = self.get_mu(queries_keys[i, 1])
            cluster_indices = self.get_cluster_indices(mu_qi, mu_kj)
            # Calculating formulas (8) and (9):
            self.mu_tilde += self.lambd * self.mu_tilde + (1 - self.lambd) * sum(cluster_indices)
            self.c_mu += self.lambd * self.c_mu + (1 - self.lambd) * len(cluster_indices)

        # Calculating formula (10)
        self.mu = self.mu_tilde / self.c_mu

    def get_mu(self, vector):
        # Returns the cluster index of the vector usually computed by some function. For simplicity, let's return a 0
        return 0

    def get_cluster_indices(self, mu_qi, mu_kj):
        # Ideally, this would retrieve the set of indices of keys that the ith query attend to.
        # However, since we don't know what these queries/keys are, let's return a placeholder set of indices
        return set([0, 1, 2, 3])
