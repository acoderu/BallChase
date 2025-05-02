class SensorGapDetector:
    """
    Enhanced sensor gap detection with predictive capabilities and heartbeat monitoring.
    Provides graduated gap levels for progressive response to sensor outages.
    """
    
    def __init__(self, sensors=None, config=None):
        self.sensors = sensors or []
        self.config = config or {}
        
        # Default gap timeout (seconds)
        self.gap_timeouts = self.config.get("gap_timeouts", {
            "default": 0.1,
            "lidar": 0.2,
            "yolo_3d": 0.15,
            "yolo_2d": 0.25,
        })
        
        # NEW: Graduated gap levels
        # Potential gap: Early warning, first sign of possible gap
        # Probable gap: Stronger indication of a gap, allows preliminary preparations
        # Confirmed gap: Definite gap, full gap handling needed
        self.gap_levels = {
            "potential": 0.5,      # Multiply default timeout by this factor for "potential" gap
            "probable": 0.8,       # Multiply default timeout by this factor for "probable" gap
            "confirmed": 1.0,      # Full timeout for confirmed gap
        }
        
        # State tracking for each sensor
        self.last_update_time = {}
        self.is_experiencing_gap = {}
        self.gap_level = {}        # NEW: Current gap level for each sensor
        self.expected_rate = {}    # NEW: Dynamically tracked rate for each sensor
        self.rate_history = {}     # NEW: History of observation rates for prediction
        
        # Initialize sensor states
        for sensor in self.sensors:
            self.last_update_time[sensor] = 0
            self.is_experiencing_gap[sensor] = False
            self.gap_level[sensor] = None
            self.expected_rate[sensor] = 10.0  # Default 10Hz assumption
            self.rate_history[sensor] = []
        
        # Statistics
        self.gap_counts = {}
        self.max_gap_duration = {}
        self.total_gap_duration = {}
        
        for sensor in self.sensors:
            self.gap_counts[sensor] = 0
            self.max_gap_duration[sensor] = 0
            self.total_gap_duration[sensor] = 0
        
        # NEW: Heartbeat timer tracking
        self.last_heartbeat_check = 0
        self.heartbeat_interval = 0.05  # Check for gaps every 50ms (20Hz)
    
    def update(self, sensor, timestamp):
        """
        Record sensor update time and calculate gap metrics.
        
        Args:
            sensor: The sensor that was updated
            timestamp: Current time in seconds
            
        Returns:
            Dictionary with sensor gap status information
        """
        if sensor not in self.sensors:
            return {}
        
        # Calculate gap duration if there was a previous update
        gap_duration = 0
        if self.last_update_time[sensor] > 0:
            gap_duration = timestamp - self.last_update_time[sensor]
            
            # NEW: Update rate history for this sensor
            if gap_duration > 0:
                rate = 1.0 / gap_duration
                self.rate_history[sensor].append(rate)
                if len(self.rate_history[sensor]) > 10:
                    self.rate_history[sensor].pop(0)
                
                # Update expected rate (moving average of last N observations)
                if len(self.rate_history[sensor]) >= 3:
                    self.expected_rate[sensor] = sum(self.rate_history[sensor]) / len(self.rate_history[sensor])
        
        # Update the statistic for maximum gap duration
        if gap_duration > self.max_gap_duration[sensor]:
            self.max_gap_duration[sensor] = gap_duration
            
        # Check if this update resolves a gap
        if self.is_experiencing_gap[sensor]:
            self.total_gap_duration[sensor] += gap_duration
            self.is_experiencing_gap[sensor] = False
            self.gap_level[sensor] = None
            
        # Record this update time
        self.last_update_time[sensor] = timestamp
        
        return {
            "sensor": sensor,
            "timestamp": timestamp,
            "gap_duration": gap_duration,
            "is_gap": self.is_experiencing_gap[sensor],
            "gap_level": self.gap_level[sensor]
        }
    
    def check_for_gaps(self, current_time):
        """
        Check all sensors for data gaps at the current time.
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            Dictionary of sensors with detected gaps and their levels
        """
        gaps = {}
        
        for sensor in self.sensors:
            # Skip sensors that haven't received any data yet
            if self.last_update_time[sensor] == 0:
                continue
                
            # Get timeout for this sensor
            timeout = self.gap_timeouts.get(sensor, self.gap_timeouts.get("default"))
            
            # Calculate time since last update
            time_since_update = current_time - self.last_update_time[sensor]
            
            # NEW: Calculate expected next update based on observed rate
            if self.expected_rate[sensor] > 0:
                expected_interval = 1.0 / self.expected_rate[sensor]
                expected_next_update = self.last_update_time[sensor] + expected_interval
                time_until_expected = expected_next_update - current_time
                
                # We're late for an expected update - potential gap
                if time_until_expected < 0:
                    # Convert to percentage of full timeout
                    gap_percentage = abs(time_until_expected) / timeout
                    
                    # Determine gap level
                    if gap_percentage >= self.gap_levels["confirmed"]:
                        # Confirmed gap - standard behavior
                        if not self.is_experiencing_gap[sensor]:
                            self.gap_counts[sensor] += 1
                            self.is_experiencing_gap[sensor] = True
                        self.gap_level[sensor] = "confirmed"
                        gaps[sensor] = {
                            "duration": time_since_update,
                            "level": "confirmed",
                            "percentage": min(gap_percentage, 2.0)  # Cap at 200%
                        }
                    elif gap_percentage >= self.gap_levels["probable"]:
                        # Probable gap - getting more concerning
                        self.gap_level[sensor] = "probable"
                        gaps[sensor] = {
                            "duration": time_since_update,
                            "level": "probable",
                            "percentage": gap_percentage
                        }
                    elif gap_percentage >= self.gap_levels["potential"]:
                        # Potential gap - early warning
                        self.gap_level[sensor] = "potential"
                        gaps[sensor] = {
                            "duration": time_since_update,
                            "level": "potential",
                            "percentage": gap_percentage
                        }
            else:
                # Fallback to original logic if we don't have rate data
                if time_since_update > timeout:
                    if not self.is_experiencing_gap[sensor]:
                        self.gap_counts[sensor] += 1
                        self.is_experiencing_gap[sensor] = True
                    
                    gaps[sensor] = {
                        "duration": time_since_update,
                        "level": "confirmed",
                        "percentage": time_since_update / timeout
                    }
        
        return gaps
    
    def heartbeat_check(self, current_time):
        """
        Performs a heartbeat check for gaps at regular intervals, independent of main update cycle.
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            Dictionary of detected gaps if enough time has passed since last check
        """
        # Only check at regular heartbeat intervals
        if current_time - self.last_heartbeat_check >= self.heartbeat_interval:
            self.last_heartbeat_check = current_time
            return self.check_for_gaps(current_time)
        return {}
    
    def get_statistics(self):
        """Get gap detection statistics."""
        stats = {}
        for sensor in self.sensors:
            stats[sensor] = {
                "gap_count": self.gap_counts[sensor],
                "max_gap_duration": self.max_gap_duration[sensor],
                "total_gap_duration": self.total_gap_duration[sensor],
                "expected_rate": self.expected_rate[sensor]
            }
        return stats
    
    def reset(self):
        """Reset gap detector state."""
        for sensor in self.sensors:
            self.last_update_time[sensor] = 0
            self.is_experiencing_gap[sensor] = False
            self.gap_level[sensor] = None
            self.gap_counts[sensor] = 0
            self.max_gap_duration[sensor] = 0
            self.total_gap_duration[sensor] = 0
            self.expected_rate[sensor] = 10.0
            self.rate_history[sensor] = []
        self.last_heartbeat_check = 0