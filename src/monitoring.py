"""
monitoring.py - Production Monitoring and Observability

Provides health checks, metrics collection, performance monitoring,
and observability features for the AIMO3 pipeline.

Features:
- Health checks for all components
- Performance metrics collection
- Error rate tracking
- Resource utilization monitoring
- Event logging for audit trails
"""

import logging
import time
import threading
import json
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def record(
        self,
        name: str,
        value: float,
        unit: str = "ms",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags for categorization
        """
        metric = PerformanceMetric(name, value, unit, tags=tags or {})
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.max_history)
            self.metrics[name].append(metric)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary with min, max, mean, count
        """
        with self.lock:
            if name not in self.metrics or len(self.metrics[name]) == 0:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            return {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics}


class HealthCheck:
    """Performs health checks on system components."""
    
    def __init__(self):
        """Initialize health check."""
        self.checks: Dict[str, Callable] = {}
        self.results: List[HealthCheckResult] = []
        self.lock = threading.Lock()
    
    def register(self, component: str, check_fn: Callable) -> None:
        """
        Register a health check function.
        
        Args:
            component: Component name
            check_fn: Function that returns HealthCheckResult
        """
        self.checks[component] = check_fn
    
    def run_all(self, timeout: float = 30.0) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Args:
            timeout: Timeout for each check in seconds
            
        Returns:
            Dictionary of component -> HealthCheckResult
        """
        results = {}
        
        for component, check_fn in self.checks.items():
            try:
                result = check_fn()
                results[component] = result
                
                with self.lock:
                    self.results.append(result)
                    # Keep only last 1000 results
                    if len(self.results) > 1000:
                        self.results = self.results[-1000:]
                        
            except Exception as e:
                logger.error(f"Health check failed for {component}: {str(e)}")
                result = HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Exception: {str(e)}"
                )
                results[component] = result
        
        return results


class ResourceMonitor:
    """Monitors system resource utilization."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory stats
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident set size
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual memory size
            "percent": memory_percent,
            "system_memory_percent": psutil.virtual_memory().percent
        }
    
    @staticmethod
    def get_cpu_usage() -> Dict[str, Any]:
        """
        Get current CPU usage.
        
        Returns:
            Dictionary with CPU stats
        """
        process = psutil.Process()
        
        return {
            "process_percent": process.cpu_percent(interval=0.1),
            "system_percent": psutil.cpu_percent(interval=0.1),
            "num_threads": process.num_threads(),
            "num_cpu_cores": psutil.cpu_count()
        }
    
    @staticmethod
    def get_disk_usage(path: str = "/") -> Dict[str, Any]:
        """
        Get disk usage.
        
        Args:
            path: Path to check
            
        Returns:
            Dictionary with disk stats
        """
        disk_info = psutil.disk_usage(path)
        
        return {
            "total_gb": disk_info.total / 1024 / 1024 / 1024,
            "used_gb": disk_info.used / 1024 / 1024 / 1024,
            "free_gb": disk_info.free / 1024 / 1024 / 1024,
            "percent": disk_info.percent
        }
    
    @staticmethod
    def get_full_report() -> Dict[str, Any]:
        """
        Get full system resource report.
        
        Returns:
            Dictionary with all resource metrics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "memory": ResourceMonitor.get_memory_usage(),
            "cpu": ResourceMonitor.get_cpu_usage(),
            "disk": ResourceMonitor.get_disk_usage()
        }


class PerformanceProfiler:
    """Context manager for profiling code performance."""
    
    def __init__(self, name: str, collector: Optional[MetricsCollector] = None):
        """
        Initialize profiler.
        
        Args:
            name: Name of the operation being profiled
            collector: Optional MetricsCollector to record metrics
        """
        self.name = name
        self.collector = collector
        self.start_time: Optional[float] = None
        self.memory_start: Optional[Dict] = None
    
    def __enter__(self):
        """Start profiling."""
        self.start_time = time.time()
        self.memory_start = ResourceMonitor.get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and record metrics."""
        if self.start_time is None:
            return
        
        duration_ms = (time.time() - self.start_time) * 1000
        memory_end = ResourceMonitor.get_memory_usage()
        
        if self.collector:
            self.collector.record(f"{self.name}_duration", duration_ms, "ms")
            if self.memory_start:
                memory_delta = memory_end["rss_mb"] - self.memory_start["rss_mb"]
                self.collector.record(f"{self.name}_memory_delta", memory_delta, "mb")
        
        logger.debug(
            f"Profiled '{self.name}': {duration_ms:.2f}ms, "
            f"memory delta: {memory_end['rss_mb'] - self.memory_start['rss_mb']:.2f}MB"
        )


class EventLogger:
    """Logs events for audit and debugging purposes."""
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize event logger.
        
        Args:
            max_events: Maximum number of events to keep in memory
        """
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.lock = threading.Lock()
    
    def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an event.
        
        Args:
            event_type: Type of event
            message: Event message
            severity: Severity level (debug, info, warning, error, critical)
            metadata: Additional metadata
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "severity": severity,
            "metadata": metadata or {}
        }
        
        with self.lock:
            self.events.append(event)
        
        # Also log using standard logger
        log_level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(log_level, f"[{event_type}] {message}")
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get logged events with optional filtering.
        
        Args:
            event_type: Filter by event type
            severity: Filter by severity
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        with self.lock:
            events = list(self.events)
        
        # Filter
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        if severity:
            events = [e for e in events if e["severity"] == severity]
        
        # Return last N events
        return events[-limit:]
