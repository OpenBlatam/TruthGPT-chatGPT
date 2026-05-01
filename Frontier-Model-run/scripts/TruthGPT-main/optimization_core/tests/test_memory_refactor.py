import asyncio
import os
import sys

# Añadir el path base al sistema para los imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.memoria_aprendizaje.sqlite_memory import SQLiteMemory

async def test_memory():
    db_path = "test_agent_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    memory = SQLiteMemory(db_path=db_path)
    user_id = "test_user_123"
    
    print("--- Testing add_message ---")
    await memory.add_message(user_id, "user", "Hola, ¿quién eres?", metadata={"session": "1"})
    await memory.add_message(user_id, "assistant", "Soy TruthGPT, tu asistente de optimización.", metadata={"tokens": 15})
    
    print("--- Testing get_history ---")
    history = await memory.get_history(user_id, limit=5)
    for msg in history:
        print(f"Role: {msg['role']}, Content: {msg['content']}, Metadata: {msg.get('metadata')}")
    
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["metadata"]["tokens"] == 15
    
    print("--- Testing clear_memory ---")
    await memory.clear_memory(user_id)
    history_after = await memory.get_history(user_id)
    assert len(history_after) == 0
    print("Memory cleared successfully.")
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
    print("Tests passed!")

if __name__ == "__main__":
    asyncio.run(test_memory())

