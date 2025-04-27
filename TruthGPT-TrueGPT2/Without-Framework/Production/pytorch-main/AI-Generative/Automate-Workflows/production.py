import torch
from torch import nn
import torch.distributed as dist
import sqlite3
import pymongo
import logging
import requests

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseQueryAutomation:
    def __init__(self, model: nn.Module, db_config: dict):
        self.model = model
        self.db_config = db_config

    def query_database(self, query: str):
        # Convert the query into token IDs (e.g., using a tokenizer)
        token_ids = self.tokenize_query(query)

        # Run the model to get logits (predictions)
        logits = self.model(token_ids)

        # Process the logits to get the best query action (e.g., top N predictions)
        top_query_action = self.decode_logits(logits)

        # Query the appropriate database based on the predicted action
        db_result = self.run_db_query(top_query_action)

        return db_result

    def tokenize_query(self, query: str):
        # Example tokenizer (replace with actual tokenization logic)
        return torch.tensor([ord(c) for c in query]).unsqueeze(0)

    def decode_logits(self, logits: torch.Tensor):
        # Convert logits to a query action (simplified version)
        predicted_token = torch.argmax(logits, dim=-1)
        return predicted_token.item()

    def run_db_query(self, action: int):
        db_type = self.db_config.get('type', 'sqlite')

        try:
            if db_type == 'sqlite':
                return self.query_sqlite(action)
            elif db_type == 'mongodb':
                return self.query_mongodb(action)
            elif db_type == 'api':
                return self.query_api(action)
            else:
                logging.error(f"Unsupported database type: {db_type}")
                return None
        except Exception as e:
            logging.error(f"Error during database query: {e}")
            return None

    def query_sqlite(self, action: int):
        try:
            conn = sqlite3.connect(self.db_config['path'])
            cursor = conn.cursor()
            query = f"SELECT * FROM data WHERE id = {action}"
            cursor.execute(query)
            result = cursor.fetchall()
            conn.close()
            logging.info(f"SQLite Query Result: {result}")
            return result
        except Exception as e:
            logging.error(f"SQLite query error: {e}")
            return None

    def query_mongodb(self, action: int):
        try:
            client = pymongo.MongoClient(self.db_config['uri'])
            db = client[self.db_config['db_name']]
            collection = db[self.db_config['collection']]
            query = {"id": action}
            result = collection.find(query)
            result = list(result)
            client.close()
            logging.info(f"MongoDB Query Result: {result}")
            return result
        except Exception as e:
            logging.error(f"MongoDB query error: {e}")
            return None

    def query_api(self, action: int):
        try:
            api_url = self.db_config['api_url']
            response = requests.get(f"{api_url}?id={action}")
            if response.status_code == 200:
                result = response.json()
                logging.info(f"API Query Result: {result}")
                return result
            else:
                logging.error(f"API error: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"API query error: {e}")
            return None


# Example DB Configuration
db_config = {
    'type': 'mongodb',  # Can be 'sqlite', 'mongodb', or 'api'
    'uri': 'mongodb://localhost:27017',
    'db_name': 'test_db',
    'collection': 'test_collection'
}

# Initialize model and automation class
model_args = ModelArgs(vocab_size=5000, dim=256, n_layers=12, max_seq_len=128, dtype="fp16")  # Example args
transformer_model = Transformer(model_args)
db_query_automation = DatabaseQueryAutomation(transformer_model, db_config)

# Example query automation
result = db_query_automation.query_database("SELECT data from table WHERE id = 123")
print(result)
