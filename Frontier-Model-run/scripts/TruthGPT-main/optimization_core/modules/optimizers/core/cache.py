
from optimization_core.core.cache_utils import MemoryCache

class ModelCache:
    def __init__(self):
        self.model_cache = MemoryCache()
    def get_optimized_model(self, model, config):
        return self.model_cache.get(str(id(model)) + str(config))
    def cache_optimized_model(self, model, config, optimized):
        self.model_cache.set(str(id(model)) + str(config), optimized)
    def get_model_stats(self):
        return self.model_cache.get_stats().to_dict()
    def clear_model_cache(self):
        self.model_cache.clear()

def create_model_cache():
    return ModelCache()

