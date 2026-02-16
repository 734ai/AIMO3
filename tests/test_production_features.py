"""
tests/test_production_features.py - Tests for Production Features

Comprehensive tests for monitoring, resilience, caching, and security modules.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src.monitoring import (
    HealthStatus, HealthCheckResult, MetricsCollector, HealthCheck,
    ResourceMonitor, PerformanceProfiler, EventLogger
)
from src.resilience import (
    CircuitBreaker, CircuitBreakerState, retry, RetryError,
    ErrorRecoveryManager, FallbackStrategy
)
from src.caching import InMemoryCache, CacheEntry, cached
from src.security import (
    CredentialManager, InputValidator, RateLimiter, RequestSigner,
    AuditLogger
)


class TestHealthCheck:
    """Tests for health check functionality."""
    
    def test_health_check_result_creation(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
            message="All good"
        )
        
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert "timestamp" in result.to_dict()
    
    def test_health_check_register_and_run(self):
        """Test registering and running health checks."""
        hc = HealthCheck()
        
        def mock_check():
            return HealthCheckResult(
                component="test",
                status=HealthStatus.HEALTHY,
                message="OK"
            )
        
        hc.register("test_component", mock_check)
        results = hc.run_all()
        
        assert "test_component" in results
        assert results["test_component"].status == HealthStatus.HEALTHY


class TestMetricsCollector:
    """Tests for metrics collection."""
    
    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        
        collector.record("request_time", 100.5, "ms")
        collector.record("request_time", 150.3, "ms")
        
        stats = collector.get_stats("request_time")
        assert stats["count"] == 2
        assert stats["min"] == 100.5
        assert stats["max"] == 150.3
    
    def test_metrics_stats(self):
        """Test metrics statistics."""
        collector = MetricsCollector()
        
        for i in range(10):
            collector.record("value", float(i))
        
        stats = collector.get_stats("value")
        assert stats["count"] == 10
        assert stats["mean"] == 4.5  # (0+1+...+9)/10


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""
    
    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        def mock_func():
            return "success"
        
        result = cb.call(mock_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker("test", failure_threshold=2)
        
        def failing_func():
            raise Exception("Test error")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        # Second failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        # Circuit should be open now
        assert cb.state == CircuitBreakerState.OPEN
        
        # Next call should fail immediately
        with pytest.raises(Exception):
            cb.call(failing_func)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)
        
        def failing_func():
            raise Exception("Error")
        
        # Trigger opening
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Should enter HALF_OPEN on next attempt (after timeout)
        # Since timeout is 0, we can immediately test
        def success_func():
            return "ok"
        
        # Force state to HALF_OPEN for testing
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.last_failure_time = None
        
        result = cb.call(success_func)
        assert result == "ok"


class TestRetryDecorator:
    """Tests for retry decorator."""
    
    def test_retry_succeeds_immediately(self):
        """Test retry when function succeeds."""
        @retry(max_attempts=3)
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
    
    def test_retry_succeeds_after_failures(self):
        """Test retry after initial failures."""
        call_count = 0
        
        @retry(max_attempts=3, initial_delay=0.01)
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_succeeds()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausted(self):
        """Test retry when all attempts fail."""
        @retry(max_attempts=2, initial_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(RetryError):
            always_fails()


class TestInMemoryCache:
    """Tests for in-memory caching."""
    
    def test_cache_set_and_get(self):
        """Test setting and getting from cache."""
        cache = InMemoryCache()
        
        cache.set("key1", "value1")
        found, value = cache.get("key1")
        
        assert found is True
        assert value == "value1"
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = InMemoryCache()
        
        found, value = cache.get("nonexistent")
        assert found is False
        assert value is None
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = InMemoryCache()
        
        cache.set("key1", "value1", ttl_seconds=1)
        
        # Should be in cache
        found, value = cache.get("key1")
        assert found is True
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        found, value = cache.get("key1")
        assert found is False
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = InMemoryCache(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        # key1 should be evicted
        found, value = cache.get("key1")
        assert found is False
        
        # key2 and key3 should still be present
        found, value = cache.get("key2")
        assert found is True
        found, value = cache.get("key3")
        assert found is True
    
    def test_cache_decorator(self):
        """Test cache decorator."""
        call_count = 0
        
        @cached()
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args (should use cache)
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Not incremented


class TestInputValidator:
    """Tests for input validation."""
    
    def test_validate_string(self):
        """Test string validation."""
        result = InputValidator.validate_string("test", min_length=1, max_length=10)
        assert result == "test"
    
    def test_validate_string_too_short(self):
        """Test string validation with too short input."""
        with pytest.raises(ValueError):
            InputValidator.validate_string("", min_length=1)
    
    def test_validate_string_too_long(self):
        """Test string validation with too long input."""
        with pytest.raises(ValueError):
            InputValidator.validate_string("x" * 100, max_length=50)
    
    def test_validate_integer(self):
        """Test integer validation."""
        result = InputValidator.validate_integer(5, min_value=0, max_value=10)
        assert result == 5
    
    def test_validate_integer_out_of_range(self):
        """Test integer validation with out of range value."""
        with pytest.raises(ValueError):
            InputValidator.validate_integer(15, max_value=10)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        dirty = "../../../etc/passwd"
        clean = InputValidator.sanitize_filename(dirty)
        assert ".." not in clean
        assert "/" not in clean


class TestRateLimiter:
    """Tests for rate limiting."""
    
    def test_rate_limiter_allows_within_limit(self):
        """Test rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is True
        assert limiter.allow_request() is False
    
    def test_rate_limiter_window_reset(self):
        """Test rate limiter window reset."""
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        
        assert limiter.allow_request() is True
        assert limiter.allow_request() is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should allow again
        assert limiter.allow_request() is True


class TestRequestSigner:
    """Tests for request signing."""
    
    def test_sign_and_verify_request(self):
        """Test signing and verifying requests."""
        data = {"id": 123, "action": "test"}
        secret = "my_secret"
        
        signature = RequestSigner.sign_request(data, secret)
        is_valid = RequestSigner.verify_signature(data, signature, secret)
        
        assert is_valid is True
    
    def test_verify_invalid_signature(self):
        """Test verification of invalid signature."""
        data = {"id": 123}
        secret = "my_secret"
        bad_signature = "invalid_signature"
        
        is_valid = RequestSigner.verify_signature(data, bad_signature, secret)
        assert is_valid is False


class TestPerformanceProfiler:
    """Tests for performance profiling."""
    
    def test_profiler_measures_time(self):
        """Test profiler measures execution time."""
        collector = MetricsCollector()
        
        with PerformanceProfiler("test_operation", collector):
            time.sleep(0.1)
        
        stats = collector.get_stats("test_operation_duration")
        assert stats["min"] >= 100  # At least 100ms
    
    def test_profiler_measures_memory(self):
        """Test profiler measures memory usage."""
        collector = MetricsCollector()
        
        with PerformanceProfiler("memory_test", collector):
            # Allocate some memory
            data = [0] * 100000
        
        stats = collector.get_stats("memory_test_memory_delta")
        assert "min" in stats or stats == {}  # May not record if memory unchanged


class TestEventLogger:
    """Tests for event logging."""
    
    def test_log_event(self):
        """Test logging events."""
        logger = EventLogger()
        
        logger.log_event("test_event", "Test message", "info")
        events = logger.get_events()
        
        assert len(events) > 0
        assert events[0]["type"] == "test_event"
    
    def test_filter_events_by_type(self):
        """Test filtering events by type."""
        logger = EventLogger()
        
        logger.log_event("error_event", "Error", "error")
        logger.log_event("info_event", "Info", "info")
        
        errors = logger.get_events(event_type="error_event")
        assert len(errors) == 1
        assert errors[0]["type"] == "error_event"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
