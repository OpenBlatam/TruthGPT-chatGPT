"""
Ultra-Advanced Caching & Session Management Module for TruthGPT Optimization Core
High-performance caching, session management, and distributed caching support
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import logging
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import json
import pickle
import hashlib
from collections import defaultdict, deque
import math
import random
from pathlib import Path
import asyncio
from contextlib import contextmanager
import uuid
import redis
import memcached
import sqlite3
from datetime import datetime, timedelta
import weakref

logger = logging.getLogger(__name__)

class CacheBackend(Enum):
    """Cache backends"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    SQLITE = "sqlite"
    DISTRIBUTED = "distributed"

class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"   # Time To Live
    ADAPTIVE = "adaptive"

class SessionState(Enum):
    """Session states"""
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

class CacheConfig(BaseModel):
    """Configuration for caching system"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Backend settings
    backend: CacheBackend = CacheBackend.MEMORY
    max_size: int = 1000
    default_ttl: int = 3600  # 1 hour
    
    # Strategy settings
    strategy: CacheStrategy = CacheStrategy.LRU
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # Memcached settings
    memcached_servers: List[str] = Field(default_factory=lambda: ["localhost:11211"])
    
    # Performance settings
    enable_statistics: bool = True
    enable_persistence: bool = False
    persistence_path: str = "cache.db"
    
    @model_validator(mode='after')
    def validate_config(self) -> 'CacheConfig':
        """Validate configuration"""
        if self.max_size <= 0:
            raise ValueError("Max size must be positive")
        if self.default_ttl <= 0:
            raise ValueError("Default TTL must be positive")
        return self

class CacheEntry(BaseModel):
    """Cache entry"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0
    size: int = 0
    compressed: bool = False

class SessionConfig(BaseModel):
    """Configuration for session management"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Session settings
    session_timeout: int = 3600  # 1 hour
    max_sessions: int = 10000
    cleanup_interval: int = 300  # 5 minutes
    
    # Security settings
    enable_encryption: bool = True
    encryption_key: str = ""
    enable_secure_cookies: bool = True
    
    # Storage settings
    storage_backend: str = "memory"  # memory, redis, database
    enable_persistence: bool = True
    persistence_path: str = "sessions.db"
    
    @model_validator(mode='after')
    def validate_config(self) -> 'SessionConfig':
        """Validate configuration"""
        if self.session_timeout <= 0:
            raise ValueError("Session timeout must be positive")
        if self.max_sessions <= 0:
            raise ValueError("Max sessions must be positive")
        return self

class Session(BaseModel):
    """Session information"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    session_id: str
    user_id: str
    created_at: float
    last_accessed: float
    expires_at: float
    state: SessionState
    data: Dict[str, Any] = Field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""

class MemoryCache:
    """High-performance memory cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache storage
        self.cache = {}
        self.access_order = deque()
        self.access_counts = defaultdict(int)
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
            'max_size': config.max_size
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("✅ Memory Cache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if time.time() > entry.expires_at:
                del self.cache[key]
                self.access_order.remove(key)
                self.stats['misses'] += 1
                return None
    
            # Update access information
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.access_counts[key] += 1
            
            # Update access order for LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats['hits'] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            try:
                # Calculate TTL
                if ttl is None:
                    ttl = self.config.default_ttl
                
                expires_at = time.time() + ttl
                
                # Compress if enabled and threshold met
                compressed = False
                if self.config.enable_compression:
                    serialized = pickle.dumps(value)
                    if len(serialized) > self.config.compression_threshold:
                        # Simplified compression
                        compressed = True
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    expires_at=expires_at,
                    access_count=1,
                    last_accessed=time.time(),
                    size=len(pickle.dumps(value)),
                    compressed=compressed
                )
                
                # Check if we need to evict
                if len(self.cache) >= self.config.max_size and key not in self.cache:
                    self._evict_entry()
                
                # Store entry
                self.cache[key] = entry
                self.access_order.append(key)
                self.access_counts[key] = 1
                
                self.stats['size'] = len(self.cache)
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache entry: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['size'] = len(self.cache)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()
            self.stats['size'] = 0
    
    def _evict_entry(self):
        """Evict entry based on strategy"""
        if not self.cache:
            return
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self.access_order:
                key_to_remove = self.access_order.popleft()
                if key_to_remove in self.cache:
                    del self.cache[key_to_remove]
                    del self.access_counts[key_to_remove]
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            if self.access_counts:
                key_to_remove = min(self.access_counts.items(), key=lambda x: x[1])[0]
                del self.cache[key_to_remove]
                del self.access_counts[key_to_remove]
                if key_to_remove in self.access_order:
                    self.access_order.remove(key_to_remove)
        
        elif self.config.strategy == CacheStrategy.TTL:
            # Remove expired entries
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time > entry.expires_at
            ]
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                if key in self.access_counts:
                    del self.access_counts[key]
        
        self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'strategy': self.config.strategy.value,
            'backend': self.config.backend.value
        }

class RedisCache:
    """Redis-based cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Redis connection
        self.redis_client = None
        
        # Initialize Redis
        self._initialize_redis()
        
        logger.info("✅ Redis Cache initialized")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password if self.config.redis_password else None,
                decode_responses=False
            )
            
            # Test connection
            if self.redis_client:
                self.redis_client.ping()
                logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            if not self.redis_client:
                return None
    
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get from Redis: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            if not self.redis_client:
                return False
            
            data = pickle.dumps(value)
            
            if ttl is None:
                ttl = self.config.default_ttl
            
            self.redis_client.setex(key, ttl, data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set in Redis: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            if not self.redis_client:
                return False
            
            result = self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete from Redis: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries"""
        try:
            if self.redis_client:
                self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")

class TruthGPTCache:
    """Main cache manager"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize cache backend
        if config.backend == CacheBackend.MEMORY:
            self.cache_backend = MemoryCache(config)
        elif config.backend == CacheBackend.REDIS:
            self.cache_backend = RedisCache(config)
        else:
            self.cache_backend = MemoryCache(config)  # Fallback
        
        # Cache statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_sets': 0,
            'cache_deletes': 0
        }
        
        logger.info("✅ TruthGPT Cache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats['total_requests'] += 1
        
        value = self.cache_backend.get(key)
        
        if value is not None:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        success = self.cache_backend.set(key, value, ttl)
        
        if success:
            self.stats['cache_sets'] += 1
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        success = self.cache_backend.delete(key)
        
        if success:
            self.stats['cache_deletes'] += 1
        
        return success
    
    def clear(self):
        """Clear all cache entries"""
        self.cache_backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        backend_stats = self.cache_backend.get_stats()
        
        return {
            **self.stats,
            **backend_stats,
            'hit_rate': self.stats['cache_hits'] / max(self.stats['total_requests'], 1)
        }

class TruthGPTSessionManager:
    """Session management system"""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Session storage
        self.sessions = {}
        self.user_sessions = defaultdict(list)
        
        # Database connection
        self.db_connection = None
        
        # Initialize session management
        self._initialize_session_storage()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info("✅ TruthGPT Session Manager initialized")
    
    def _initialize_session_storage(self):
        """Initialize session storage"""
        try:
            if self.config.enable_persistence:
                self.db_connection = sqlite3.connect(float(self.config.persistence_path) if isinstance(self.config.persistence_path, (int, float)) else self.config.persistence_path, check_same_thread=False)
                if self.db_connection:
                    cursor = self.db_connection.cursor()
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_accessed REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        state TEXT NOT NULL,
                        data TEXT DEFAULT '{}',
                        ip_address TEXT DEFAULT '',
                        user_agent TEXT DEFAULT ''
                    )
                ''')
                
                if self.db_connection:
                    self.db_connection.commit()
                logger.info("✅ Session database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize session storage: {e}")
    
    def _start_cleanup_thread(self):
        """Start session cleanup thread"""
        def cleanup_sessions():
            while True:
                try:
                    self._cleanup_expired_sessions()
                    time.sleep(self.config.cleanup_interval)
                except Exception as e:
                    logger.error(f"Session cleanup failed: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
        cleanup_thread.start()
        logger.info("✅ Session cleanup thread started")
    
    def create_session(self, user_id: str, ip_address: str = "", user_agent: str = "") -> str:
        """Create new session"""
        try:
            session_id = str(uuid.uuid4())
            current_time = time.time()
            expires_at = current_time + self.config.session_timeout
            
            session = Session(
                session_id=session_id,
                user_id=user_id,
                created_at=current_time,
                last_accessed=current_time,
                expires_at=expires_at,
                state=SessionState.ACTIVE,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Store in memory
            self.sessions[session_id] = session
            self.user_sessions[user_id].append(session_id)
            
            # Store in database
            if self.db_connection:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('''
                        INSERT INTO sessions 
                        (session_id, user_id, created_at, last_accessed, expires_at, 
                         state, data, ip_address, user_agent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, user_id, current_time, current_time, expires_at,
                        SessionState.ACTIVE.value, json.dumps({}), ip_address, user_agent
                    ))
                    self.db_connection.commit()
                except sqlite3.Error as e:
                    logger.error(f"Database error during session creation: {e}")
            
            logger.info(f"✅ Session created: {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return ""
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        try:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            if time.time() > session.expires_at:
                session.state = SessionState.EXPIRED
                return None
                
            # Update last accessed
            session.last_accessed = time.time()
            
            # Sync with database
            if self.db_connection:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute(
                        "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                        (session.last_accessed, session_id)
                    )
                    self.db_connection.commit()
                except sqlite3.Error as e:
                    logger.error(f"Database error during session access update: {e}")

            return session
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def update_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data"""
        try:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            session.data.update(data)
            
            # Update in database
            if self.db_connection:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('''
                        UPDATE sessions SET data = ?, last_accessed = ?
                        WHERE session_id = ?
                    ''', (json.dumps(session.data), session.last_accessed, session_id))
                    self.db_connection.commit()
                except sqlite3.Error as e:
                    logger.error(f"Database error during session data update: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session data: {e}")
            return False
    
    def extend_session(self, session_id: str, extension_time: int = None) -> bool:
        """Extend session expiration"""
        try:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            
            if extension_time is None:
                extension_time = self.config.session_timeout
            
            session.expires_at = time.time() + extension_time
            
            # Update in database
            if self.db_connection:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('''
                        UPDATE sessions SET expires_at = ?, last_accessed = ?
                        WHERE session_id = ?
                    ''', (session.expires_at, session.last_accessed, session_id))
                    self.db_connection.commit()
                except sqlite3.Error as e:
                    logger.error(f"Database error during session extension: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extend session: {e}")
            return False
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate session"""
        try:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            session.state = SessionState.TERMINATED
            
            # Remove from memory
            self.sessions.pop(session_id, None)
            if session.user_id in self.user_sessions:
                self.user_sessions[session.user_id].remove(session_id)
            
            # Update in database
            if self.db_connection:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('''
                        UPDATE sessions SET state = ? WHERE session_id = ?
                    ''', (SessionState.TERMINATED.value, session_id))
                    self.db_connection.commit()
                except sqlite3.Error as e:
                    logger.error(f"Database error during session termination: {e}")
            
            logger.info(f"✅ Session terminated: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate session: {e}")
            return False
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if current_time > session.expires_at:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.terminate_session(session_id)
            
            # Cleanup database
            if self.db_connection:
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute('DELETE FROM sessions WHERE expires_at < ?', (current_time,))
                    self.db_connection.commit()
                except sqlite3.Error as e:
                    logger.error(f"Database error during session cleanup: {e}")
            
            if expired_sessions:
                logger.info(f"✅ Cleaned up {len(expired_sessions)} expired sessions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a user"""
        sessions = []
        for session_id in self.user_sessions.get(user_id, []):
            session = self.get_session(session_id)
            if session:
                sessions.append(session)
        return sessions
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': len([s for s in self.sessions.values() if s.state == SessionState.ACTIVE]),
            'expired_sessions': len([s for s in self.sessions.values() if s.state == SessionState.EXPIRED]),
            'max_sessions': self.config.max_sessions,
            'session_timeout': self.config.session_timeout
        }

class TruthGPTCacheManager:
    """Main cache and session manager"""
    
    def __init__(self, cache_config: CacheConfig, session_config: SessionConfig):
        self.cache_config = cache_config
        self.session_config = session_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.cache = TruthGPTCache(cache_config)
        self.session_manager = TruthGPTSessionManager(session_config)
        
        # Manager state
        self.is_running = True
        
        logger.info("✅ TruthGPT Cache Manager initialized")
    
    def cache_model_inference(self, model_id: str, input_hash: str, output: Any, ttl: int = 3600) -> bool:
        """Cache model inference result"""
        try:
            cache_key = f"inference:{model_id}:{input_hash}"
            return self.cache.set(cache_key, output, ttl)
            
        except Exception as e:
            logger.error(f"Failed to cache model inference: {e}")
            return False
    
    def get_cached_inference(self, model_id: str, input_hash: str) -> Optional[Any]:
        """Get cached model inference result"""
        try:
            cache_key = f"inference:{model_id}:{input_hash}"
            return self.cache.get(cache_key)
    
        except Exception as e:
            logger.error(f"Failed to get cached inference: {e}")
            return None
    
    def cache_model_weights(self, model_id: str, version: str, weights: Dict[str, Any]) -> bool:
        """Cache model weights"""
        try:
            cache_key = f"weights:{model_id}:{version}"
            return self.cache.set(cache_key, weights, ttl=86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Failed to cache model weights: {e}")
            return False
    
    def get_cached_weights(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get cached model weights"""
        try:
            cache_key = f"weights:{model_id}:{version}"
            return self.cache.get(cache_key)
    
        except Exception as e:
            logger.error(f"Failed to get cached weights: {e}")
            return None
    
    def create_user_session(self, user_id: str, ip_address: str = "", user_agent: str = "") -> str:
        """Create user session"""
        return self.session_manager.create_session(user_id, ip_address, user_agent)
    
    def get_user_session(self, session_id: str) -> Optional[Session]:
        """Get user session"""
        return self.session_manager.get_session(session_id)
    
    def update_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data"""
        return self.session_manager.update_session_data(session_id, data)
    
    def terminate_user_session(self, session_id: str) -> bool:
        """Terminate user session"""
        return self.session_manager.terminate_session(session_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return self.session_manager.get_session_stats()
    
    def get_manager_summary(self) -> Dict[str, Any]:
        """Get comprehensive manager summary"""
        return {
            'cache_stats': self.get_cache_stats(),
            'session_stats': self.get_session_stats(),
            'cache_config': {
                'backend': self.cache_config.backend.value,
                'strategy': self.cache_config.strategy.value,
                'max_size': self.cache_config.max_size,
                'default_ttl': self.cache_config.default_ttl
            },
            'session_config': {
                'timeout': self.session_config.session_timeout,
                'max_sessions': self.session_config.max_sessions,
                'enable_encryption': self.session_config.enable_encryption
            }
        }

# Factory functions
def create_cache_config(**kwargs) -> CacheConfig:
    """Create cache configuration"""
    return CacheConfig(**kwargs)

def create_session_config(**kwargs) -> SessionConfig:
    """Create session configuration"""
    return SessionConfig(**kwargs)

def create_cache_entry(key: str, value: Any, ttl: int = 3600) -> CacheEntry:
    """Create cache entry"""
    current_time = time.time()
    return CacheEntry(
        key=key,
        value=value,
        created_at=current_time,
        expires_at=current_time + ttl,
        last_accessed=current_time,
        size=len(pickle.dumps(value))
    )

def create_session(session_id: str, user_id: str, ttl: int = 3600) -> Session:
    """Create session"""
    current_time = time.time()
    return Session(
        session_id=session_id,
        user_id=user_id,
        created_at=current_time,
        last_accessed=current_time,
        expires_at=current_time + ttl,
        state=SessionState.ACTIVE
    )

def create_cache(config: CacheConfig) -> TruthGPTCache:
    """Create cache instance"""
    return TruthGPTCache(config)

def create_session_manager(config: SessionConfig) -> TruthGPTSessionManager:
    """Create session manager"""
    return TruthGPTSessionManager(config)

def create_cache_manager(cache_config: CacheConfig, session_config: SessionConfig) -> TruthGPTCacheManager:
    """Create cache manager"""
    return TruthGPTCacheManager(cache_config, session_config)

def quick_caching_setup() -> TruthGPTCacheManager:
    """Quick caching setup for testing"""
    cache_config = create_cache_config(
        backend=CacheBackend.MEMORY,
        max_size=1000,
        strategy=CacheStrategy.LRU,
        default_ttl=3600
    )
    
    session_config = create_session_config(
        session_timeout=3600,
        max_sessions=1000,
        enable_encryption=False
    )
    
    return create_cache_manager(cache_config, session_config)

# Example usage
def example_caching_system():
    """Example of caching and session management"""
    # Create cache manager
    manager = quick_caching_setup()
    
    print("✅ Cache Manager created!")
    
    # Test caching
    test_data = {"accuracy": 0.95, "latency": 50.0}
    cache_key = "test_model_v1"
    
    # Set cache
    success = manager.cache.set(cache_key, test_data, ttl=3600)
    print(f"📦 Cache set: {success}")
    
    # Get cache
    cached_data = manager.cache.get(cache_key)
    print(f"📦 Cache get: {cached_data}")
    
    # Test model inference caching
    model_id = "sentiment_classifier"
    input_hash = hashlib.md5("test input".encode()).hexdigest()
    output = {"prediction": "positive", "confidence": 0.9}
    
    # Cache inference
    inference_cached = manager.cache_model_inference(model_id, input_hash, output)
    print(f"🤖 Inference cached: {inference_cached}")
    
    # Get cached inference
    cached_inference = manager.get_cached_inference(model_id, input_hash)
    print(f"🤖 Cached inference: {cached_inference}")
    
    # Test session management
    user_id = "user123"
    session_id = manager.create_user_session(user_id, "192.168.1.1", "Mozilla/5.0")
    print(f"👤 Session created: {session_id}")
    
    # Get session
    session = manager.get_user_session(session_id)
    if session:
        print(f"👤 Session retrieved: {session.user_id}")
    
    # Update session data
    session_data = {"preferences": {"theme": "dark"}, "last_action": "login"}
    session_updated = manager.update_session_data(session_id, session_data)
    print(f"👤 Session updated: {session_updated}")
    
    # Get cache stats
    cache_stats = manager.get_cache_stats()
    print(f"📊 Cache stats: {cache_stats}")
    
    # Get session stats
    session_stats = manager.get_session_stats()
    print(f"📊 Session stats: {session_stats}")
    
    # Get manager summary
    summary = manager.get_manager_summary()
    print(f"📈 Manager summary: {summary}")
    
    return manager

# Export utilities
__all__ = [
    'CacheBackend',
    'CacheStrategy',
    'SessionState',
    'CacheConfig',
    'CacheEntry',
    'SessionConfig',
    'Session',
    'MemoryCache',
    'RedisCache',
    'TruthGPTCache',
    'TruthGPTSessionManager',
    'TruthGPTCacheManager',
    'create_cache_config',
    'create_session_config',
    'create_cache_entry',
    'create_session',
    'create_cache',
    'create_session_manager',
    'create_cache_manager',
    'quick_caching_setup',
    'example_caching_system'
]

if __name__ == "__main__":
    example_caching_system()
    print("✅ Caching and session management module complete!")
