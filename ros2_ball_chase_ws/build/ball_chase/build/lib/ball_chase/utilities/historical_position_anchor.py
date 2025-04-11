class HistoricalPositionAnchor:
    """
    Maintains a history of recent positions to provide anchoring
    during sensor gaps for stationary objects.
    """
    
    def __init__(self, max_history_length=60):
        # Position history storage
        self.position_history = []
        self.timestamp_history = []
        self.max_history_length = max_history_length
        
        # Computed anchor points
        self.current_anchor = None
        self.anchor_confidence = 0.0  # 0.0-1.0 scale
        
        # Motion state tracking
        self.last_motion_state = "unknown"
    
    def update(self, position, timestamp, motion_state):
        """
        Add a new position to the history and update the anchor if needed.
        
        Args:
            position: [x, y, z] position
            timestamp: Current time
            motion_state: Current motion state of the object
            
        Returns:
            Tuple of (anchor_position, confidence) where confidence is 0.0-1.0
        """
        # Store new position with timestamp
        self.position_history.append(list(position))
        self.timestamp_history.append(timestamp)
        
        # Trim history if needed
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
            self.timestamp_history.pop(0)
        
        # Update anchor based on motion state
        self.last_motion_state = motion_state
        
        if motion_state in ["stationary", "long_stationary"]:
            # For stationary objects, compute anchor as recent average
            # Use more positions for long-stationary objects
            if motion_state == "long_stationary":
                # Use up to last 30 positions (3 seconds at 10Hz)
                history_length = min(30, len(self.position_history))
                confidence = min(1.0, history_length / 30.0)
            else:
                # Use up to last 10 positions (1 second at 10Hz)
                history_length = min(10, len(self.position_history))
                confidence = min(0.7, history_length / 10.0)
            
            # Calculate centroid of recent positions
            if history_length > 0:
                recent_positions = self.position_history[-history_length:]
                
                # Calculate weighted average, giving more weight to more recent positions
                anchor = [0.0, 0.0, 0.0]
                total_weight = 0.0
                
                for i, pos in enumerate(recent_positions):
                    # Use linear weighting: newer positions get higher weights
                    weight = i + 1
                    total_weight += weight
                    
                    # Accumulate weighted position
                    anchor[0] += pos[0] * weight
                    anchor[1] += pos[1] * weight
                    anchor[2] += pos[2] * weight
                
                # Normalize by total weight
                self.current_anchor = [coord / total_weight for coord in anchor]
                self.anchor_confidence = confidence
            else:
                self.current_anchor = None
                self.anchor_confidence = 0.0
        else:
            # For moving objects, gradually reduce anchor confidence
            # We might still want a weak anchor for brief stops
            if self.anchor_confidence > 0:
                self.anchor_confidence = max(0.0, self.anchor_confidence - 0.2)
                if self.anchor_confidence <= 0:
                    self.current_anchor = None
        
        return (self.current_anchor, self.anchor_confidence)
    
    def get_anchor(self):
        """
        Get the current position anchor and confidence.
        
        Returns:
            Tuple of (anchor_position, confidence)
        """
        return (self.current_anchor, self.anchor_confidence)
    
    def reset(self):
        """Reset the anchor state."""
        self.position_history = []
        self.timestamp_history = []
        self.current_anchor = None
        self.anchor_confidence = 0.0
        self.last_motion_state = "unknown"