#!/usr/bin/env python3

"""
Performance Metrics Collector
============================

A utility class to collect and report comprehensive performance metrics
for the BallChase vision system.
"""

import time
import psutil
from collections import deque
import numpy as np
import threading

class PerformanceMetrics:
    """
    Collects detailed performance metrics for vision processing nodes.
    Provides a unified interface for tracking:
    - FPS
    - CPU usage (accurate)
    - Memory usage
    - Processing latency
    - Detection quality statistics
    """
    
    def __init__(self, window_size=20):
        """Initialize the performance metrics collector."""
        # Timing metrics
        self.fps_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.last_update_time = time.time()
        self.current_fps = 0.0
        self.avg_fps = 0.0
        
        # CPU metrics
        self.cpu_usage_history = deque(maxlen=window_size)
        self.current_cpu = 0.0
        self.process = psutil.Process()
        
        # Memory metrics
        self.memory_usage_history = deque(maxlen=window_size)
        self.current_memory = 0.0
        
        # Detection quality metrics
        self.quality_ratings = deque(maxlen=window_size*2)  # Longer history for quality
        self.valid_points_history = deque(maxlen=window_size)
        
        # Cache performance
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_hit_rate = 0.0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Background monitoring
        self.should_monitor = False
        self.monitor_thread = None

    def start_monitoring(self, interval=1.0):
        """Start background monitoring of system resources."""
        self.should_monitor = True
        self.monitor_thread = threading.Thread(
            target=self._background_monitor, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.should_monitor = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _background_monitor(self, interval):
        """Background thread to monitor system resources."""
        while self.should_monitor:
            try:
                # Update CPU and memory
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory().percent
                proc_cpu = self.process.cpu_percent() / psutil.cpu_count()
                
                with self.lock:
                    self.current_cpu = cpu
                    self.cpu_usage_history.append(cpu)
                    self.current_memory = mem
                    self.memory_usage_history.append(mem)
            except:
                pass
                
            # Sleep for the specified interval
            time.sleep(interval)
    
    def update_fps(self, frame_count=1):
        """Update FPS calculation with the given frame count."""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_update_time
            
            if elapsed > 0.1:  # Only update if enough time has passed
                instantaneous_fps = frame_count / elapsed
                self.fps_history.append(instantaneous_fps)
                self.current_fps = instantaneous_fps
                
                # Update average FPS
                if self.fps_history:
                    self.avg_fps = sum(self.fps_history) / len(self.fps_history)
                    
                # Reset timer
                self.last_update_time = current_time
                
            return self.current_fps
    
    def update_latency(self, latency_ms):
        """Record processing latency in milliseconds."""
        with self.lock:
            self.latency_history.append(latency_ms)
    
    def update_detection_quality(self, quality_rating, valid_points=0):
        """Record detection quality metrics."""
        with self.lock:
            self.quality_ratings.append(quality_rating)
            if valid_points > 0:
                self.valid_points_history.append(valid_points)
    
    def update_cache_stats(self, hit=False):
        """Update cache hit/miss statistics."""
        with self.lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                
            total = self.cache_hits + self.cache_misses
            if total > 0:
                self.cache_hit_rate = (self.cache_hits / total) * 100.0
    
    def get_stats(self):
        """Get a comprehensive statistics report."""
        with self.lock:
            stats = {
                # FPS stats
                "fps": round(self.current_fps, 2),
                "avg_fps": round(self.avg_fps, 2),
                
                # CPU and memory
                "cpu": round(self.current_cpu, 1),
                "memory": round(self.current_memory, 1),
                
                # Latency stats if available
                "avg_latency_ms": round(np.mean(self.latency_history), 2) if self.latency_history else 0,
                
                # Detection quality
                "avg_quality_points": round(np.mean(self.valid_points_history), 1) if self.valid_points_history else 0,
                
                # Cache performance
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(self.cache_hit_rate, 1),
                
                # Timestamp
                "timestamp": time.time()
            }
            
            # Calculate quality ratings if available
            if self.quality_ratings:
                quality_counts = {
                    "good": 0,
                    "fair": 0, 
                    "poor": 0
                }
                
                for rating in self.quality_ratings:
                    if rating in quality_counts:
                        quality_counts[rating] += 1
                
                total = len(self.quality_ratings)
                if total > 0:
                    for rating in quality_counts:
                        quality_counts[rating] = (quality_counts[rating] / total) * 100.0
                
                stats["quality_distribution"] = {
                    rating: round(pct, 1) for rating, pct in quality_counts.items()
                }
            
            return stats
            
    def get_formatted_stats(self):
        """Get a formatted string with key statistics."""
        stats = self.get_stats()
        
        # Format the core metrics
        formatted = (
            f"FPS: {stats['fps']:.1f} (avg: {stats['avg_fps']:.1f}), "
            f"CPU: {stats['cpu']:.1f}%, "
            f"RAM: {stats['memory']:.1f}%, "
            f"Cache: {stats['cache_hit_rate']:.1f}%"
        )
        
        # Add latency if available
        if stats['avg_latency_ms'] > 0:
            formatted += f", Latency: {stats['avg_latency_ms']:.1f}ms"
        
        # Add quality if available
        if 'quality_distribution' in stats:
            quality = stats['quality_distribution']
            formatted += f", Quality: {quality['good']:.0f}% good, {quality['fair']:.0f}% fair"
        
        return formatted