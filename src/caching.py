"""
caching.py - Production-Grade Caching System

Provides multiple caching strategies including in-memory, with TTL,
LRU eviction, and persistent cache support.

Features:
- In-memory caching with configurable max size
- Time-to-live (TTL) for cache entries
- LRU eviction policy
- Cache statistics and hit/miss rates
- Thread-safe operations
"""

import logging
import time
import threading
import json
import hashlib
import pickle
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path


logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheEntry:
    """Represents a single cache entry."""
    
    def __init__(self, value: Any, ttl_seconds: Optional[int] = None):
        """
        Initialize cache entry.
        
        Args:
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        self.value = value
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class CacheStats:
    """Cache statistics."""
    
    def __init__(self):
        """Initialize statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate(),
            "total_requests": self.hits + self.misses
        }


class InMemoryCache:
    """
    Thread-safe in-memory cache with LRU eviction and TTL support.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: Optional[int] = None
    ):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl_seconds: Default time-to-live for entries
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.Lock()
    
    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """
        Generate cache key from function arguments.
        
        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        # Convert args and kwargs to a hashable representation
        key_parts = [func_name]
        
        try:
            for arg in args:
                if isinstance(arg, (str, int, float, bool, type(None))):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(str(id(arg)))
            
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            key_str = "|".join(key_parts)
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {str(e)}")
            return func_name  # Fallback to function name
        
        # Hash to keep key size bounded
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (found, value)
        """
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return False, None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return False, None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.stats.hits += 1
            
            return True, entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL for this entry (overrides default)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        entry = CacheEntry(value, ttl)
        
        with self.lock:
            # If key exists, remove it first (to update position)
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            
            # Evict oldest entry if over capacity
            if len(self.cache) > self.max_size:
                removed_key, _ = self.cache.popitem(last=False)
                self.stats.evictions += 1
                logger.debug(f"Evicted cache entry: {removed_key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            return self.stats
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        with self.lock:
            initial_size = len(self.cache)
            
            expired_keys = [
                k for k, v in self.cache.items()
                if v.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.stats.expirations += 1
            
            removed = initial_size - len(self.cache)
            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired cache entries")
            
            return removed


def cached(
    cache: Optional[InMemoryCache] = None,
    ttl_seconds: Optional[int] = None
) -> Callable:
    """
    Decorator to cache function results.
    
    Args:
        cache: Cache instance (creates default if None)
        ttl_seconds: Time-to-live for cached values
        
    Returns:
        Decorated function
    """
    _cache = cache or InMemoryCache()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            key = _cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            found, value = _cache.get(key)
            if found:
                logger.debug(f"Cache hit for {func.__name__}")
                return value
            
            # Not in cache, compute value
            logger.debug(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            
            # Store in cache
            _cache.set(key, result, ttl_seconds)
            
            return result
        
        wrapper.cache = _cache  # Expose cache for testing
        return wrapper
    
    return decorator


class PersistentCache:
    """
    Persistent cache backed by JSON files.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache key."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """
        Get value from persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (found, value)
        """
        with self.lock:
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                return False, None
            
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                # Check expiration
                if 'expires_at' in data:
                    expires_at = datetime.fromisoformat(data['expires_at'])
                    if datetime.now() > expires_at:
                        cache_path.unlink()
                        return False, None
                
                return True, data['value']
            except Exception as e:
                logger.error(f"Failed to read cache: {str(e)}")
                return False, None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set value in persistent cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
        """
        with self.lock:
            cache_path = self._get_cache_path(key)
            
            try:
                data = {
                    'key': key,
                    'value': value,
                    'created_at': datetime.now().isoformat()
                }
                
                if ttl_seconds:
                    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
                    data['expires_at'] = expires_at.isoformat()
                
                with open(cache_path, 'w') as f:
                    json.dump(data, f, default=str)
            except Exception as e:
                logger.error(f"Failed to write cache: {str(e)}")
    
    def clear(self) -> None:
        """Clear all persistent cache entries."""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file: {str(e)}")
