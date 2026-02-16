"""
resilience.py - Error Handling, Retries, and Resilience Patterns

Provides production-grade error handling, retry logic with exponential backoff,
circuit breaker pattern, and fallback strategies.

Features:
- Automatic retry with exponential backoff
- Circuit breaker pattern for cascading failures
- Timeout handling
- Fallback strategies
- Error categorization and recovery
"""

import logging
import time
import threading
from typing import Callable, TypeVar, Optional, Any, Dict, List
from functools import wraps
from enum import Enum
from datetime import datetime, timedelta
import random


logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorCategory(Enum):
    """Categorizes errors for handling decisions."""
    TRANSIENT = "transient"          # Temporary, retry likely to succeed
    PERMANENT = "permanent"          # Won't succeed with retry
    UNKNOWN = "unknown"              # Can't determine, default to transient


class RetryError(Exception):
    """Raised when all retries are exhausted."""
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        """
        Initialize retry error.
        
        Args:
            message: Error message
            last_exception: The last exception encountered
        """
        super().__init__(message)
        self.last_exception = last_exception


class CircuitBreakerState(Enum):
    """States for circuit breaker pattern."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, reject requests
    HALF_OPEN = "half_open"    # Testing if recovered


class CircuitBreaker:
    """
    Implements circuit breaker pattern for fault tolerance.
    
    Transitions:
    - CLOSED -> OPEN: After failure_threshold failures
    - OPEN -> HALF_OPEN: After timeout duration
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: After any failure
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes before closing circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.Lock()
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise Exception(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retrying in {self._time_until_retry()}s"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _time_until_retry(self) -> int:
        """Get seconds until retry is allowed."""
        if self.last_failure_time is None:
            return 0
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return max(0, int(self.recovery_timeout - elapsed))
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed (recovered)")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        with self.lock:
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened after failure in HALF_OPEN state")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after "
                        f"{self.failure_count} failures"
                    )


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[type]] = None
) -> Callable:
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_exceptions: List of exception types to retry on
        
    Returns:
        Decorated function
    """
    if retryable_exceptions is None:
        retryable_exceptions = [Exception]
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {str(e)}"
                        )
                        raise RetryError(
                            f"Failed after {max_attempts} attempts: {str(e)}",
                            last_exception
                        )
                    
                    # Calculate delay with exponential backoff
                    delay = initial_delay * (exponential_base ** (attempt - 1))
                    delay = min(delay, max_delay)
                    
                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            # Should not reach here, but just in case
            raise RetryError(f"Failed after {max_attempts} attempts", last_exception)
        
        return wrapper
    
    return decorator


def timeout(seconds: float) -> Callable:
    """
    Decorator to add timeout to function execution.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} exceeded timeout of {seconds}s")
            
            # Set signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        return wrapper
    
    return decorator


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_handle(self, exception: Exception) -> bool:
        """
        Check if this strategy can handle the exception.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if this strategy can handle it
        """
        raise NotImplementedError
    
    def handle(self, exception: Exception, context: Dict[str, Any]) -> Any:
        """
        Handle the exception.
        
        Args:
            exception: Exception to handle
            context: Context information
            
        Returns:
            Recovery value or raises exception
        """
        raise NotImplementedError


class FallbackStrategy(ErrorRecoveryStrategy):
    """Uses a fallback function when primary fails."""
    
    def __init__(
        self,
        exception_type: type,
        fallback_fn: Callable[..., T]
    ):
        """
        Initialize fallback strategy.
        
        Args:
            exception_type: Type of exception to handle
            fallback_fn: Function to call as fallback
        """
        self.exception_type = exception_type
        self.fallback_fn = fallback_fn
    
    def can_handle(self, exception: Exception) -> bool:
        """Check if exception is of handled type."""
        return isinstance(exception, self.exception_type)
    
    def handle(self, exception: Exception, context: Dict[str, Any]) -> Any:
        """
        Execute fallback function.
        
        Args:
            exception: Original exception
            context: Context with 'args' and 'kwargs' keys
            
        Returns:
            Result from fallback function
        """
        try:
            args = context.get('args', ())
            kwargs = context.get('kwargs', {})
            return self.fallback_fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback strategy failed: {str(e)}")
            raise


class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.strategies: List[ErrorRecoveryStrategy] = []
    
    def register_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """
        Register an error recovery strategy.
        
        Args:
            strategy: Strategy to register
        """
        self.strategies.append(strategy)
    
    def register_fallback(
        self,
        exception_type: type,
        fallback_fn: Callable
    ) -> None:
        """
        Register a fallback function for an exception type.
        
        Args:
            exception_type: Exception type to handle
            fallback_fn: Fallback function
        """
        self.register_strategy(FallbackStrategy(exception_type, fallback_fn))
    
    def handle(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Find and execute a recovery strategy.
        
        Args:
            exception: Exception to handle
            context: Context information
            
        Returns:
            Recovery value or None if no strategy found
            
        Raises:
            Exception: If no strategy can handle the exception
        """
        context = context or {}
        
        for strategy in self.strategies:
            if strategy.can_handle(exception):
                try:
                    return strategy.handle(exception, context)
                except Exception as e:
                    logger.error(f"Recovery strategy failed: {str(e)}")
                    continue
        
        # No strategy found, re-raise
        raise exception
