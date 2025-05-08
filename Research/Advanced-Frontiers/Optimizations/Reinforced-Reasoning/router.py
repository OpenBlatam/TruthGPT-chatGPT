class MRKLSystem:
    def __init__(self, modules, llm):
        self.modules = modules  # Dictionary of expert modules
        self.router = Router(llm)  # BERT-based classification layer
        
    def route_query(self, input_text):
        module_scores = self.router(input_text) 
        return self.modules[module_scores.argmax()]
