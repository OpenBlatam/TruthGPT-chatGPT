"""
OpenClaw Memory -- Core Memory (Working Memory).

Provides a persistent 'working memory' block that the agent can read and write.
This follows the MemGPT/Letta pattern where agents have a fixed-size context 
window for vital information (User Profile, Agent Persona) that persists 
across sessions but is distinct from the massive Archival/Vector memory.
"""

import json
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class CoreMemory:
    """
    Persistent Working Memory for agents.
    
    Stores two main blocks:
    - persona: The agent's own rules, identity, and current goals.
    - human: Facts about the user that the agent should never forget.
    """

    def __init__(self, db_path: str = "agent_core_memory.db") -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS core_memory (
                    user_id TEXT PRIMARY KEY,
                    persona TEXT DEFAULT '',
                    human   TEXT DEFAULT ''
                )
            """)
            conn.commit()

    async def get_core(self, user_id: str) -> Dict[str, str]:
        """Retrieve the core memory blocks for a user asynchronously."""
        return await asyncio.to_thread(self._get_core_sync, user_id)

    def _get_core_sync(self, user_id: str) -> Dict[str, str]:
        """Synchronous helper for get_core."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT persona, human FROM core_memory WHERE user_id = ?", 
                    (user_id,)
                ).fetchone()
                
                if row:
                    return {"persona": row["persona"], "human": row["human"]}
                return {"persona": "No specific persona defined.", "human": "No specific user info recorded."}
        except Exception as e:
            logger.error(f"Error reading core memory: {e}")
            return {"persona": "Error", "human": "Error"}

    async def update_block(self, user_id: str, block: str, content: str) -> bool:
        """Update a specific block (persona or human) asynchronously."""
        return await asyncio.to_thread(self._update_block_sync, user_id, block, content)

    def _update_block_sync(self, user_id: str, block: str, content: str) -> bool:
        """Synchronous helper for update_block."""
        if block not in ["persona", "human"]:
            return False
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Upsert logic
                conn.execute(f"""
                    INSERT INTO core_memory (user_id, {block}) 
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET {block}=excluded.{block}
                """, (user_id, content))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating core memory block {block}: {e}")
            return False

    async def append_to_block(self, user_id: str, block: str, content: str) -> bool:
        """Append a line to a specific block asynchronously."""
        core = await self.get_core(user_id)
        current = core.get(block, "")
        new_content = f"{current}\n- {content}".strip()
        return await self.update_block(user_id, block, new_content)

    async def get_formatted_context(self, user_id: str) -> str:
        """Returns the core memory formatted for prompt injection asynchronously."""
        core = await self.get_core(user_id)
        return (
            "\n=== CORE WORKING MEMORY ===\n"
            f"[AGENT PERSONA]:\n{core['persona']}\n\n"
            f"[USER PROFILE]:\n{core['human']}\n"
            "===========================\n"
        )

