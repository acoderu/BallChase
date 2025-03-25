#!/usr/bin/env python3

import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.lifecycle.node import LifecycleState
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

class EnhancedFusionNode(LifecycleNode):
    def __init__(self):
        super().__init__('enhanced_fusion_node')
        self.get_logger().info("======xxxxx Enhanced Fusion Lifecycle Node Starting ======")
        
        # Core tracking variables
        self.start_time = time.time()
        self.transform_available = False
        self.transform_checks = 0
        self.transform_successes = 0
        self.transform_failures = 0
        self.transform_confirmed = False
        self.is_ready = False
        self.reference_frame = "camera_frame"
        
        # Add lifecycle-specific variables
        self.transform_retry_count = 0
        self.max_transform_retries = 10
        self.transform_retry_timer = None

    def on_configure(self, state):
        """
        First lifecycle transition - handles the initial setup phase
        Equivalent to PHASE 1 in your original code
        """
        self.get_logger().info("Lifecycle on_configure: Initializing transform system")
        
        # Initialize ONLY the transform system here
        self.init_transform_system()
        
        # Start a timer to check transform availability
        self.transform_retry_count = 0
        self.transform_retry_timer = self.create_timer(
            5.0, 
            self.check_transform_availability_for_activation,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        # Return success to indicate configuration completed
        return TransitionCallbackReturn.SUCCESS

    def check_transform_availability_for_activation(self, force_complete=False):
        """
        Timer callback to check if transform is available
        When transform is confirmed, cancels itself and triggers activation
        """
        self.transform_checks += 1
        transform_available = self.check_transform_availability()
        
        if transform_available or force_complete:
            # Cancel this timer as we don't need more checks
            self.transform_retry_timer.cancel()
            self.transform_retry_timer = None
            
            # Load configuration (PHASE 2)
            self.load_configuration()
            
            # Initialize state tracking (PHASE 3)
            self.init_state_tracking()
            self.init_sensor_synchronization()
            
            # Trigger transition to active state
            self.trigger_transition(LifecycleState.ACTIVATING)
            return
        
        # If transform still not available, increment retry counter
        self.transform_retry_count += 1
        if self.transform_retry_count >= self.max_transform_retries:
            self.get_logger().error("Transform unavailable after maximum retries - failing activation")
            # Force completion to allow activation to fail gracefully
            self.check_transform_availability_for_activation(force_complete=True)

    def on_activate(self, state):
        """
        Second lifecycle transition - handles the final setup and starts operations
        Equivalent to PHASES 4-8 in your original code
        """
        if not self.transform_confirmed:
            self.get_logger().error("Activation attempted without confirmed transform - failing")
            return TransitionCallbackReturn.FAILURE
        
        self.get_logger().info("Lifecycle on_activate: Transform confirmed - proceeding with initialization")
        
        # PHASE 4: Set up publishers
        self.setup_publishers()
        
        # PHASE 5: Initialize diagnostics
        self.init_diagnostics()
        
        # PHASES 6: Set up subscriptions (only now that transform is available)
        self.setup_subscriptions()
        
        # PHASE 7: Initialize filter with defaults
        self.initialize_filter_with_defaults()
        
        # PHASE 8: Set up processing timers
        self.setup_timers()
        
        # Mark as ready
        self.is_ready = True
        self.get_logger().info("Initialization complete - node is ready")
        
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """
        Handle deactivation (stopping operations but keeping configuration)
        """
        # Cancel timers
        if hasattr(self, 'filter_timer') and self.filter_timer:
            self.filter_timer.cancel()
        if hasattr(self, 'status_timer') and self.status_timer:
            self.status_timer.cancel()
        if hasattr(self, 'diagnostics_timer') and self.diagnostics_timer:
            self.diagnostics_timer.cancel()
        
        self.is_ready = False
        self.get_logger().info("Node deactivated - operations stopped")
        
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """
        Handle cleanup (returning to unconfigured state)
        """
        # Reset internal state
        self.transform_available = False
        self.transform_confirmed = False
        
        # Reset other state variables as needed
        self.get_logger().info("Node cleaned up - returned to unconfigured state")
        
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        """
        Handle shutdown request
        """
        self.get_logger().info("Node shutting down")
        
        return TransitionCallbackReturn.SUCCESS
    
    def check_transform_availability(self):
        """
        Check if transform is available without trying to fix anything.
        Returns True if transform is available, False otherwise.
        """
        # Original transform check code, but with any logging level adjustments needed
        # ...existing code...
        
        # Return transform status
        if self.transform_available:
            # Once we've confirmed the transform is available, we don't need to keep checking
            if not self.transform_confirmed and self.transform_successes >= 2:
                self.transform_confirmed = True
                self.get_logger().info("Transform availability confirmed permanently")
        
        return self.transform_available
    
    def publish_status(self):
        """Publish and log brief status information."""
        # Skip if not active
        if not self.is_ready or self.get_current_state().id != LifecycleState.ACTIVE:
            return
            
        # ... existing publish_status implementation ...
    
    # Keep all existing method implementations intact
    # ...existing methods for init_transform_system, load_configuration, etc...


def main(args=None):
    """Main function to start the node."""
    rclpy.init(args=args)
    
    # Create the lifecycle node
    node = EnhancedFusionNode()
    
    # Create a separate thread for the executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        # Trigger configuration
        node.trigger_transition(LifecycleState.CONFIGURING)
        
        # Wait for executor thread to finish (it never will, but this keeps the main thread alive)
        executor_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        # Trigger shutdown
        node.trigger_transition(LifecycleState.SHUTTINGDOWN)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
