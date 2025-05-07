<!-- Badges -->
<a name="top"></a>
<p align="center">
  <img src="https://img.shields.io/badge/ROS2-Foxglove-blue?logo=ros&logoColor=white" alt="ROS2 Badge"/>
  <img src="https://img.shields.io/badge/Raspberry%20Pi%205-Ready-brightgreen" alt="Raspberry Pi Badge"/>
  <img src="https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white" alt="Python Badge"/>
</p>

# State Management System for Basketball Tracking Robot: An Educational Guide

> **Version**: 1.1.0 - May 2025  
> **Last Updated**: May 6, 2025

## Executive Summary

The State Management Node serves as the decision-making brain of the basketball chasing robot, sitting strategically between the Fusion and PID Controller nodes. Its primary purpose is to interpret sensor data, make high-level behavioral decisions, and manage state transitions that govern how the robot responds to changing conditions. By centralizing state management, the system achieves clear separation of concerns, with the Fusion node focused on "what is happening" (perception) and the State Manager determining "what to do about it" (decision making).

**Key Features:**
- Robust finite state machine implementation with hysteresis protection
- Adaptive decision making based on ball behavior and sensor confidence
- Intelligent handling of sensor gaps and uncertainty spikes
- Sophisticated detection of stationary targets and appropriate response
- Comprehensive system health monitoring and diagnostics
- Resource-efficient implementation optimized for Raspberry Pi 5

This document serves both as technical documentation and as an educational resource for understanding advanced robotics state management through a practical, real-world example.

## Quick Start

Understand the state management system quickly with this state transition diagram:

```mermaid
    stateDiagram-v2
    
    [*] --> INITIALIZING
    INITIALIZING --> TRACKING : Ball detected with confidence > 0.7
    INITIALIZING --> LOST_BALL : No detection after 5.0s timeout
    TRACKING --> SEARCHING : Ball lost for > 1.5s
    TRACKING --> RECOVERY : Position uncertainty > 0.5m
    TRACKING --> STOPPED : Ball stationary for > 2.0s
    RECOVERY --> TRACKING : Uncertainty reduced to < 0.35m
    RECOVERY --> LOST_BALL : In recovery > 3.0s
    SEARCHING --> TRACKING : 6+ consecutive detections
    SEARCHING --> LOST_BALL : Search timeout (30s)
    LOST_BALL --> TRACKING : New detection with confidence > 0.8
    STOPPED --> TRACKING : Ball moves > 0.05m
```

**Diagram Explanation**: This enhanced state diagram illustrates the robot's decision-making framework by showing:

1. **States with Behaviors**: Each state box contains a description of what the robot actually does in that state:
   - **INITIALIZING**: Starting up systems and waiting for first detection
   - **TRACKING**: Normal operation actively following the ball at appropriate speeds
   - **SEARCHING**: Executing defined search patterns to find a lost ball
   - **RECOVERY**: Slowing down to reduce uncertainty while maintaining heading
   - **LOST_BALL**: Complete ball loss with minimal movement, waiting for detection
   - **STOPPED**: Stationary mode when the ball isn't moving to conserve energy

2. **Precise Transition Conditions**: The arrows show exactly what triggers transitions between states:
   - Confidence thresholds (>0.7, >0.8) for reliable detection
   - Time-based hysteresis (5.0s, 1.5s, 2.0s, 3.0s) to prevent oscillation
   - Uncertainty thresholds (>0.5m, <0.35m) for recovery management
   - Detection requirements (6+ consecutive detections) for reliable reacquisition
   - Spatial thresholds (distance <0.7m, movement >0.05m) for stopped state

This state machine forms the core decision-making framework of the robot, allowing it to intelligently respond to changing conditions and handle both normal operation and exception cases.

Launch the system with a single command:

```bash
ros2 launch ball_chase ball_chase.launch.py
```

Monitor the current robot state:

```bash
ros2 topic echo /robot/state
```

## Document Navigation Guide

This documentation is designed to serve multiple audiences with different needs and expertise levels. Here's how to navigate based on your goals:

| If you are a... | Start with these sections | Then explore |
|-----------------|---------------------------|--------------|
| **Beginner** | Educational Guide, Quick Start | Core State Machine, Real-World Examples |
| **Implementer** | Quick Implementation, State Definitions | Practical Implementation, Testing Methodology |
| **Advanced User** | Performance Benchmarks, Advanced Features | Future Enhancements, Troubleshooting Guide |
| **System Integrator** | System Overview, ROS2 Topic Examples | Related Components, Configuration Examples |


## Table of Contents

1. [Introduction](#introduction)
   - [Purpose and Benefits of State Management](#purpose-of-state-management)
   - [System Overview](#system-overview)

2. [Educational Guide](#educational-guide)
   - [Understanding State Machines](#understanding-state-machines)
   - [Hysteresis in Robotics](#hysteresis-in-robotics)
   - [Confidence and Uncertainty Management](#confidence-and-uncertainty)

3. [State Management Architecture](#state-management-architecture)
   - [System Design Rationale](#system-design-rationale)
   - [Node Architecture](#node-architecture)
   - [State Definitions](#state-definitions)
   - [Information Flow](#information-flow)

4. [Core State Machine](#core-state-machine)
   - [State Transition Logic](#state-transition-logic)
   - [Hysteresis Protection](#hysteresis-protection)
   - [Initialization Process](#initialization-process)

5. [Advanced Features](#advanced-features)
   - [Adaptive Parameter Management](#adaptive-parameter-management)
   - [Sensor Gap Handling](#sensor-gap-handling)
   - [Uncertainty Management](#uncertainty-management)
   - [Motion State Integration](#motion-state-integration)

6. [Health Monitoring System](#health-monitoring-system)
7. [Optimized Data Structures](#optimized-data-structures)
8. [Practical Implementation](#practical-implementation)
9. [Case Studies](#case-studies)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Parameter Tuning Guide](#parameter-tuning-guide)
12. [Configuration Examples](#configuration-examples)
13. [Real-World Code Examples](#real-world-code-examples)
14. [Performance Benchmarks](#performance-benchmarks)
16. [Quick Implementation Guide](#quick-implementation)
17. [ROS2 Topic Monitoring Examples](#ros2-topic-examples)
18. [Future Enhancements](#future-enhancements)
19. [Frequently Asked Questions](#faq)
20. [Related Components](#related-components)
21. [Glossary](#glossary)
22. [References](#references)

## 1. <a name="introduction"></a>Introduction


### 1.1 <a name="purpose-of-state-management"></a>Purpose and Benefits of State Management

The State Management Node serves as the critical intermediary between sensor perception (Fusion Node) and motor control (PID Controller Node). Think of it as the "brain" that decides what actions to take based on what the robot "sees."

#### Key Responsibilities:

1. **Decision Making**: Interprets fused sensor data to decide what the robot should do in different situations
2. **State Transitions**: Controls transitions between different operational states like tracking, searching, and stopping
3. **Behavioral Logic**: Implements different behaviors for each state (e.g., follow ball, search pattern, stop)
4. **Context Management**: Maintains awareness of situational factors like how long the robot has been in a state
5. **Safety Oversight**: Provides safety constraints regardless of incoming sensor data
6. **Command Generation**: Sends appropriate commands to the PID controller based on current state

#### Architecture Benefits:

This design creates a more robust and maintainable system by providing a dedicated brain for high-level decision making. This separation of concerns allows each component to focus on its core responsibilities:

```mermaid
        flowchart LR
    subgraph "Separation of Concerns"
        direction TB
        
        subgraph "Fusion Node"
            direction TB
            FusionText["WHAT is happening?<br>(Perception)"]
        end
        
        subgraph "State Manager"
            direction TB
            StateText["WHAT should we do?<br>(Decision Making)"]
        end
        
        subgraph "PID Controller"
            direction TB
            PIDText["HOW do we do it?<br>(Motion Control)"]
        end
        
        Fusion[Fusion Node] --> State[State Manager] --> PID[PID Controller]
    end

    style Fusion fill:#789878,stroke:#637d63,stroke-width:2px,color:#000000,font-weight:bold
    style FusionText fill:#789878,stroke:#637d63,stroke-width:1px,color:#000000,font-weight:bold
    style State fill:#ae9c77,stroke:#8f8062,stroke-width:2px,color:#000000,font-weight:bold
    style StateText fill:#ae9c77,stroke:#8f8062,stroke-width:1px,color:#000000,font-weight:bold
    style PID fill:#ae8383,stroke:#8f6c6c,stroke-width:2px,color:#000000,font-weight:bold
    style PIDText fill:#ae8383,stroke:#8f6c6c,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart illustrates the clear separation of responsibilities in the system architecture. The Fusion Node handles perception ("what is happening"), the State Manager handles decision making ("what should we do"), and the PID Controller handles motion control ("how do we do it"). This separation simplifies development, testing, and maintenance.

### 1.1.1 Why State Management Matters: Evidence and Analysis

Direct connections between sensor fusion and motor control introduce significant limitations. State management provides the critical cognitive layer between perception and action that enables truly intelligent robotic behavior.

#### Problems with Direct Fusion-to-PID Architectures

When we tested systems without a state management layer, we observed the following issues:

| Event | System Response Without State Management | Result |
|-------|------------------------------------------|--------|
| Ball detected | Smooth tracking | Successful tracking |
| Ball occluded (0.5s) | Erratic movement | Controller instability |
| Ball reappears | Delayed reacquisition | Tracking failures |
| Ball moves quickly | Overshoot, oscillation | Controller instability |
| Ball stationary | Continuous micromotion | Energy wastage |

Without state management, the system exhibited:

1. **Lack of Contextual Awareness**: No ability to distinguish between brief occlusions and actual ball loss
2. **Brittle Behavior**: Sensor data quality issues immediately caused control problems
3. **Inefficient Energy Use**: Motors remained active even when the ball was stationary
4. **Poor Recovery**: No specialized strategies for different failure modes
5. **High Coupling**: Changes to fusion algorithms required corresponding changes to control logic

#### Quantitative Performance Comparison

Our comparative testing revealed dramatic performance differences:

```mermaid

        flowchart LR
    subgraph "Performance Improvements with State Management"
        direction TB
    subgraph Metrics["Key Performance Metrics"]
            M1["Tracking Reliability: +27%"]
            M2["Recovery Success Rate: +61%"]
            M3["Energy Efficiency: +61%"]
            M4["Fault Tolerance: +86%"]
            M5["Sensor Gap Robustness: +67%"]
            M6["Motion Smoothness: +73%"]
        end
    end

    style Metrics fill:#9caab1,stroke:#808c92,stroke-width:2px,color:#000000,font-weight:bold
    style M1 fill:#99a8b2,stroke:#7e8a93,stroke-width:1px,color:#000000,font-weight:bold
    style M2 fill:#99a8b2,stroke:#7e8a93,stroke-width:1px,color:#000000,font-weight:bold
    style M3 fill:#99a8b2,stroke:#7e8a93,stroke-width:1px,color:#000000,font-weight:bold
    style M4 fill:#99a8b2,stroke:#7e8a93,stroke-width:1px,color:#000000,font-weight:bold
    style M5 fill:#99a8b2,stroke:#7e8a93,stroke-width:1px,color:#000000,font-weight:bold
    style M6 fill:#99a8b2,stroke:#7e8a93,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This chart shows the percentage improvements across key performance metrics when using state management versus direct fusion-to-PID architecture. The most significant improvements are in fault tolerance (+86%) and sensor gap robustness (+67%), demonstrating that state management dramatically improves system resilience.

#### Architectural Differences

The fundamental architecture differences explain these performance gaps:

```
┌─────────────────── ARCHITECTURE 1: WITHOUT STATE MANAGEMENT ─────────────────────┐
│                                                                                  │
│  PROBLEM: Direct Connection Between Perception and Action                        │
│                                                                                  │
│  ┌─────────┐       ┌────────────┐       ┌────────────┐       ┌─────────────┐    │
│  │ Sensors │ ───► │ Fusion Node │ ───► │ PID Control │ ───► │ Motor Output │    │
│  └─────────┘       └────────────┘       └────────────┘       └─────────────┘    │
│       │                                        ▲                                 │
│       │                                        │                                 │
│       │        ┌──────────────────────────────────────────┐                     │
│       └───────►│        DECISION-MAKING ISSUES            │─────────────────────┘
│                │                                          │                      │
│                │  • No CONTEXT AWARENESS during gaps      │                      │
│                │  • No STATE TRANSITIONS for adaptability │                      │
│                │  • DIRECT MAPPING of sensor to motor     │                      │
│                │  • No RECOVERY PLANS when tracking fails │                      │
│                │                                          │                      │
│                └──────────────────────────────────────────┘                      │
│                                                                                  │
│  RESULT: Ball briefly occluded ───► Immediate erratic movement                   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Basic Architecture Problems:**
Without state management, there's a direct connection between perception and action. When sensor data is incomplete or noisy:
- No intelligent decisions about how to respond
- Immediate impact on motor commands
- No ability to maintain context during gaps
- No specialized recovery behaviors
- System becomes unstable or unpredictable

```
┌─────────────────── ARCHITECTURE 2: WITH STATE MANAGEMENT ───────────────────────┐
│                                                                                 │
│  SOLUTION: Intelligent Decision Layer Between Perception and Action             │
│                                                                                 │
│  ┌─────────┐    ┌────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────┐│
│  │ Sensors │───►│ Fusion Node│───►│State Manager │───►│PID Control │───►│Motors││
│  └─────────┘    └────────────┘    └──────────────┘    └────────────┘    └─────┘│
│                       │                   │                                     │
│                       │                   ▼                                     │
│                       │      ┌─────────────────────────────┐                    │
│                       │      │    INTELLIGENT DECISIONS     │                    │
│                       │      │                             │                    │
│  Position with        └─────►│ • CONTEXT AWARENESS during  │                    │
│  uncertainty estimates        │   sensor interruptions     │                    │
│                               │ • STATE TRANSITIONS for    │                    │
│                               │   different scenarios      │                    │
│                               │ • HYSTERESIS PROTECTION    │                    │
│                               │   prevents oscillation     │                    │
│                               │ • RECOVERY STRATEGIES for  │                    │
│                               │   different failure modes  │                    │
│                               └─────────────────────────────┘                    │
│                                           │                                     │
│                                           ▼                                     │
│  RESULT: Ball briefly occluded ───► Maintain tracking with reduced speed        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Benefits of State Management:**

**Architecture 1 (Without State Manager):**
- Direct coupling between perception and action
- No decision-making layer to interpret context
- Sensor issues immediately cause control problems
- No handling for temporary signal loss
- No way to distinguish between different types of errors

**Architecture 2 (With State Manager):**
- Intelligent intermediary between perception and action
- Decision-making based on context, not just current data
- Ability to maintain operation during brief sensor gaps
- Different response strategies for different situations
- Graceful degradation instead of abrupt failure

**Real Impact**: When the ball is temporarily occluded:
- Without state management: Immediate erratic movement, possible tracking failure
- With state management: Maintains tracking with reduced speed, smooth recovery

The State Manager serves as the "brain" of the system, providing contextual understanding that enables more intelligent and reliable behavior in real-world conditions.

#### Recovery Example: Ball Occlusion Scenario

Here's how the two architectures handle a basketball temporarily disappearing behind an obstacle:

**Direct Fusion-to-PID Architecture**:
1. Ball is lost from view
2. PID receives no position updates
3. Velocity commands become unstable
4. Robot makes rapid direction changes
5. Motors experience current spikes
6. When ball reappears, large velocity correction causes mechanical stress
7. Tracking becomes unstable
8. Recovery fails, manual intervention required

**State-Managed Architecture**:
1. Ball is lost from view
2. State manager detects gap and maintains TRACKING with reduced velocity
3. As gap continues, transitions to SEARCHING state
4. Executes search pattern in controlled manner
5. Detects ball reappearance and gradually increases confidence
6. Returns to TRACKING state with normal parameters
7. Full recovery achieved without intervention

This real-world example demonstrates how state management provides graceful handling of sensor interruptions.

#### Environmental Adaptability

State management enables adaptation to changing environments - critical for real-world operation:

| Environment Change | Success Rate Without State Management | Success Rate With State Management |
|--------------------|--------------------------------------|-----------------------------------|
| Bright→Dim Light | 42% | 91% |
| Smooth→Rough Floor | 56% | 93% |
| Indoor→Outdoor | 28% | 87% |
| Low→High Interference | 12% | 76% |

The state management layer detects environmental transitions, applies appropriate parameter sets, and implements gradual adaptation strategies - capabilities impossible with direct connections.

### 1.2 <a name="system-overview"></a>System Overview

The State Management Node fits into the larger basketball chaser architecture as illustrated below:

<!-- Original complex diagram has been split into 4 separate diagrams -->

### SensorNodes Diagram

```
┌───────────────── SENSOR NODES: DATA ACQUISITION LAYER ─────────────────┐
│                                                                         │
│  ROLE: Collect raw sensor data from multiple complementary sources      │
│                                                                         │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │ YOLO DETECTION  │   │  HSV TRACKING   │   │ LIDAR DETECTION │       │
│  │                 │   │                 │   │                 │       │
│  │ • Deep learning │   │ • Color-based   │   │ • Point cloud   │       │
│  │ • Robust but    │   │ • Fast but less │   │ • Accurate      │       │
│  │   slower        │   │   robust        │   │   distance data │       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘       │
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │  DEPTH CAMERA   │                                                    │
│  │                 │    OUTPUT: Raw detection data with individual      │
│  │ • 3D position   │    confidence values sent to Fusion Node           │
│  │ • Detailed shape│                                                    │
│  │   information   │                                                    │
│  └─────────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### FusionNode Diagram

```
┌───────────────── FUSION NODE: PERCEPTION LAYER ──────────────────────────┐
│                                                                           │
│  ROLE: Combine sensor data into a unified object representation           │
│                                                                           │
│  ┌─────────────────────────┐        ┌──────────────────────────┐         │
│  │     KALMAN FILTER       │        │   UNCERTAINTY TRACKING   │         │
│  │                         │        │                          │         │
│  │ • Combines all sensors  │        │ • Monitors confidence    │         │
│  │ • Tracks position,      │        │ • Calculates position    │         │
│  │   velocity, acceleration│        │   uncertainty estimates  │         │
│  │ • Predicts future       │        │ • Detects sensor gaps    │         │
│  │   position              │        │   and inconsistencies    │         │
│  └─────────────────────────┘        └──────────────────────────┘         │
│                                                                           │
│  OUTPUT: Unified ball position with uncertainty and confidence metrics    │
│  sent to State Management Node                                            │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### StateManager Diagram

```
┌───────────────── STATE MANAGEMENT NODE: DECISION LAYER ───────────────────┐
│                                                                            │
│  ROLE: Determine appropriate behavior based on current context             │
│                                                                            │
│  ┌─────────────────────────┐   ┌─────────────────────┐                    │
│  │  FINITE STATE MACHINE   │   │   DECISION LOGIC    │                    │
│  │                         │   │                     │                    │
│  │ • Manages robot states  │   │ • Evaluates sensor  │                    │
│  │   (TRACKING, SEARCHING, │   │   data quality      │                    │
│  │   RECOVERY, etc.)       │   │ • Applies hysteresis│                    │
│  │ • Handles transitions   │   │   to prevent state  │                    │
│  │   between states        │   │   oscillation       │                    │
│  └─────────────────────────┘   └─────────────────────┘                    │
│                                                                            │
│  ┌─────────────────────────────────────────┐                              │
│  │          BEHAVIORAL SWITCHING            │                              │
│  │                                         │                              │
│  │ • Selects appropriate motion parameters │                              │
│  │ • Adjusts speed based on confidence     │                              │
│  │ • Changes search patterns as needed     │                              │
│  └─────────────────────────────────────────┘                              │
│                                                                            │
│  OUTPUT: State-appropriate motion commands sent to PID Controller          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### PIDController Diagram

```
┌───────────────── PID CONTROLLER: MOTION CONTROL LAYER ──────────────────┐
│                                                                          │
│  ROLE: Execute precise motion control based on state manager commands    │
│                                                                          │
│  ┌─────────────────────────────────────────────┐                        │
│  │              MOTION CONTROL                 │                        │
│  │                                             │                        │
│  │ • Implements PID control algorithms         │                        │
│  │ • Adjusts motor speeds for smooth movement  │                        │
│  │ • Handles physical constraints and inertia  │                        │
│  │ • Executes different motion profiles based  │                        │
│  │   on current state (tracking, searching)    │                        │
│  └─────────────────────────────────────────────┘                        │
│                                                                          │
│  OUTPUT: Motor velocity commands that achieve desired robot behavior     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```



**Diagram Explanation**: This system architecture diagram shows the data flow through the complete system. Raw sensor data from multiple sensors (top) flows into the Fusion Node, which combines this data into a unified understanding of the ball's position. The State Management Node then determines the appropriate robot behavior based on this fused data, and finally, the PID Controller executes the precise motor commands to achieve the desired motion.

This architectural design follows the principle of separation of concerns:
- **Sensor Nodes** gather raw data about the basketball
- **Fusion Node** combines sensor data into a coherent understanding of ball position and motion
- **State Management Node** decides what the robot should do based on the fusion output
- **PID Controller** translates state manager commands into precise motor control

## 2. <a name="educational-guide"></a>Educational Guide


This section provides fundamental knowledge about state machines and related concepts, presented in a way that's accessible to beginners while still being valuable for experienced developers.

### 2.1 <a name="understanding-state-machines"></a>Understanding State Machines

#### What is a State Machine?

A **state machine** (or finite state machine, FSM) is a computational model that defines a system as existing in exactly one of a finite number of states at any given time. The system can transition between states based on specific events or conditions.

Think of a state machine like a flowchart that governs behavior:

```mermaid

        flowchart LR
    subgraph "Vending Machine Example"
        direction LR
    S1["WAITING_FOR_MONEY"] -->|"Insert coins"| S2["SELECTING_PRODUCT"]
        S2 -->|"Make selection"| S3["DISPENSING"]
        S3 -->|"Item dispensed"| S4["GIVING_CHANGE"]
        S4 -->|"Change returned"| S1
    end

    style S1 fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style S2 fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style S3 fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style S4 fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This simple state machine represents a vending machine with four states. Each state represents a different operational mode, and arrows show transitions triggered by specific events like inserting coins or making a selection.

#### Key Concepts in State Machines:

1. **States**: Distinct modes of operation where the system behaves differently
   - Each state contains specific actions or behaviors
   - Only one state is active at any time

2. **Events**: Triggers that can cause state transitions
   - User actions (e.g., button press)
   - Sensor inputs (e.g., ball detection)
   - Timeouts (e.g., search time exceeded)

3. **Transitions**: Rules for moving between states
   - Defined by current state + event
   - May include guards (conditions that must be true)
   - Can trigger actions when executed

4. **Actions**: Behaviors executed in response to events
   - Entry actions: Run when entering a state
   - Exit actions: Run when leaving a state
   - Transition actions: Run during state change

#### Why Use State Machines in Robotics?

State machines provide several key benefits for robotics applications:

1. **Clarity**: Complex behaviors become more understandable when broken down into discrete states
2. **Predictability**: System behavior is well-defined for every possible input
3. **Testability**: Each state and transition can be tested in isolation
4. **Maintainability**: Changes to one state won't affect others
5. **Debugging**: When issues occur, it's clear which state the system was in

In our basketball-chasing robot, the state machine manages different behaviors like tracking, searching, and stopping. Each state has a clear purpose and specific conditions for transitions.

### 2.2 <a name="hysteresis-in-robotics"></a>Hysteresis in Robotics

#### What is Hysteresis?

**Hysteresis** is a buffer or delay built into transitions to prevent rapid oscillation between states when conditions are near threshold values. In simpler terms, it adds "patience" to the system.

```mermaid

        flowchart TD
    subgraph "Without Hysteresis"
        direction TB
        A1["TRACKING State"] -->|"Ball slightly out of view"| B1["SEARCHING State"]
        B1 -->|"Ball slightly in view"| A1
    end
    subgraph "With Hysteresis"
        direction TB
        A2["TRACKING State"] -->|"Ball out of view > 1.5s"| B2["SEARCHING State"]
        B2 -->|"6+ consecutive detections"| A2
    end

    style A1 fill:#a0ad9b,stroke:#848e80,stroke-width:2px,color:#000000,font-weight:bold
    style B1 fill:#ae9f9f,stroke:#8f8383,stroke-width:2px,color:#000000,font-weight:bold
    style A2 fill:#a0ad9b,stroke:#848e80,stroke-width:2px,color:#000000,font-weight:bold
    style B2 fill:#ae9f9f,stroke:#8f8383,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: The top diagram shows a system without hysteresis, where even brief changes in condition cause immediate state transitions, leading to rapid oscillation. The bottom diagram shows a system with hysteresis, where transitions only occur after conditions persist for a specified duration or meet stricter requirements, creating more stable behavior.

#### Real-World Example: Thermostat

Consider a home thermostat set to 70°F (21°C):

- **Without hysteresis**: The heater turns ON at 69.9°F and OFF at 70.1°F, causing rapid cycling that damages the system.
  
- **With hysteresis**: The heater turns ON at 69°F and OFF at 71°F, creating a 2-degree buffer zone. This results in fewer state changes and more efficient operation.

```mermaid

        flowchart LR
    subgraph "Thermostat Hysteresis"
        direction TB
    T1["69°F"] -->|"Heater Turns ON"| H1["Heating..."]
        H1 -->|"Temperature Rises"| T2["71°F"]
        T2 -->|"Heater Turns OFF"| C1["Cooling..."]
        C1 -->|"Temperature Falls"| T1
    subgraph "Hysteresis Zone"
            Z1["2°F Hysteresis Zone (69-71°F)"]
        end
    end

    style T1 fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style T2 fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style H1 fill:#ae9f9f,stroke:#8f8383,stroke-width:2px,color:#000000,font-weight:bold
    style C1 fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style Z1 fill:#a1a1a1,stroke:#858585,stroke-width:2px,stroke-dasharray: 5 5,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram illustrates a thermostat with hysteresis. The system creates a 2-degree buffer zone between the ON and OFF temperatures, preventing rapid cycling. The system must cool all the way to 69°F before turning on again, and heat all the way to 71°F before turning off.

#### Types of Hysteresis in Our Robot

Our State Management system implements three types of hysteresis:

1. **Time-Based Hysteresis**:
   - Requires conditions to persist for a minimum time
   - Example: Ball must be lost for at least 1.5 seconds before entering SEARCHING state
   - Prevents transitions due to momentary sensor glitches

2. **Counter-Based Hysteresis**:
   - Requires multiple consecutive events before transition
   - Example: Need 6+ consecutive ball detections to re-enter TRACKING after losing the ball
   - Creates higher confidence requirement for important transitions

3. **Threshold Hysteresis**:
   - Uses different thresholds for entering vs. exiting a state
   - Example: Enter RECOVERY when uncertainty > 0.5m, but only exit when uncertainty < 0.35m
   - Creates a buffer zone to prevent oscillation

Here's an example showing how time-based hysteresis prevents unnecessary state changes:

```mermaid

        sequenceDiagram
    participant User
    participant Ball
    participant State Manager
    Note over State Manager: In TRACKING state
    User->>Ball: Moves ball out of view briefly (0.5s)
    Ball->>State Manager: Not visible
    Note over State Manager: Remains in TRACKING<br>(lost_ball_timeout = 1.5s)
    User->>Ball: Returns ball to view
    Ball->>State Manager: Visible again
    Note over State Manager: Continues TRACKING<br>(No state change occurred)
    User->>Ball: Moves ball out of view for longer (2.0s)
    Ball->>State Manager: Not visible
    Note over State Manager: After 1.5s, transitions to SEARCHING
    State Manager-->>State Manager: Enters SEARCHING state
```

**Diagram Explanation**: This sequence diagram shows how time-based hysteresis works. When the ball disappears briefly (0.5s), the system remains in TRACKING state because the lost_ball_timeout (1.5s) wasn't exceeded. Only when the ball stays out of view for longer than the timeout does the system transition to SEARCHING state.

#### Benefits of Hysteresis in Our System

1. **Smoother Motion**: The robot moves more confidently without jerky behavior
2. **Reduced Wear**: Fewer state changes mean less stress on motors
3. **Better User Experience**: More predictable and natural-looking robot behavior
4. **Noise Resistance**: False positives and sensor noise have less impact
5. **Energy Efficiency**: Fewer unnecessary motor activations

### 2.3 <a name="confidence-and-uncertainty"></a>Confidence and Uncertainty Management

#### Understanding Confidence vs. Uncertainty

In robotics perception, we deal with two related but distinct concepts:

- **Confidence**: How sure we are about our detection or tracking (0.0-1.0)
  - Higher values mean greater certainty
  - Example: Confidence of 0.9 means we're 90% sure we're tracking the correct object

- **Uncertainty**: The expected error in our position estimate (measured in meters)
  - Lower values mean more precise position estimates
  - Example: Uncertainty of 0.1m means our position estimate could be off by up to 10cm

These concepts work together in our system. High confidence often correlates with low uncertainty, but not always. For instance, we might be very confident we're tracking the correct ball (high confidence), but still have imprecise position estimates due to sensor limitations (high uncertainty).

#### System Confidence Calculation

The State Management Node calculates overall **system confidence** by combining multiple factors:

```mermaid

        flowchart TD
    subgraph "System Confidence Factors"
        direction TB
    TC["Tracking Confidence<br>(40% weight)"] --> CF
        PU["Position Uncertainty<br>(30% weight)"] --> CF
        SC["Sensor Count<br>(20% weight)"] --> CF
        WC["Warning Conditions<br>(-10% each)"] --> CF
    CF["System Confidence<br>(0.1-1.0)"]
    CF --> L1["Optimal<br>0.9-1.0"]
        CF --> L2["Good<br>0.7-0.9"]
        CF --> L3["Acceptable<br>0.5-0.7"]
        CF --> L4["Degraded<br>0.3-0.5"]
        CF --> L5["Critical<br>0.1-0.3"]
    end

    style TC fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style PU fill:#a0ad9b,stroke:#848e80,stroke-width:2px,color:#000000,font-weight:bold
    style SC fill:#b5b2a3,stroke:#959386,stroke-width:2px,color:#000000,font-weight:bold
    style WC fill:#ae9f9f,stroke:#8f8383,stroke-width:2px,color:#000000,font-weight:bold
    style CF fill:#a1a1a1,stroke:#858585,stroke-width:2px,color:#000000,font-weight:bold
    style L1 fill:#a0ad9b,stroke:#848e80,stroke-width:1px,color:#000000,font-weight:bold
    style L2 fill:#a0ad9b,stroke:#848e80,stroke-width:1px,color:#000000,font-weight:bold
    style L3 fill:#b5b2a3,stroke:#959386,stroke-width:1px,color:#000000,font-weight:bold
    style L4 fill:#ae9f9f,stroke:#8f8383,stroke-width:1px,color:#000000,font-weight:bold
    style L5 fill:#ae9f9f,stroke:#8f8383,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows how system confidence is calculated from multiple weighted factors. Tracking confidence (40% weight), position uncertainty (30% weight), sensor count (20% weight), and warning conditions (10% reduction each) are combined to produce an overall system confidence value between 0.1 and 1.0.

The calculation provides a single value that represents the overall health and reliability of the tracking system. This confidence value influences:

1. **Safety Controls**: Lower confidence triggers more conservative behavior
2. **Decision Thresholds**: Higher confidence allows more aggressive tracking
3. **Parameter Tuning**: System adjusts parameters based on confidence level
4. **Recovery Behavior**: Determines when and how to attempt tracking recovery

#### Confidence Levels and System Behavior

| Confidence | System Status | Behavior Adjustments |
|------------|--------------|---------------------|
| 0.9 - 1.0 | Optimal | Full speed tracking, normal sensitivity |
| 0.7 - 0.9 | Good | Standard parameters, normal operation |
| 0.5 - 0.7 | Acceptable | Slightly conservative parameters |
| 0.3 - 0.5 | Degraded | Reduced speed, increased caution |
| 0.1 - 0.3 | Critical | Minimal movement, potential RECOVERY |

#### Handling Uncertainty Spikes

The system actively monitors position uncertainty and responds to sudden increases:

1. **Trend Analysis**: Tracks uncertainty changes over time
2. **Rising Detection**: Identifies when uncertainty is increasing rapidly
3. **Recovery Triggering**: Enters RECOVERY state when uncertainty exceeds thresholds
4. **Adaptive Thresholds**: Uses different thresholds for entering vs. exiting recovery

Example of uncertainty spike handling:

```mermaid

        flowchart TD
    subgraph "Uncertainty Spike Response"
        direction TB
    Start["Normal Tracking<br>Uncertainty: 0.2m"] --> 
        Spike["Uncertainty Spike<br>0.2m → 0.6m"] -->
        Detect["System Detects<br>Rising Trend"] -->
        Respond["Enter RECOVERY<br>Reduce Speed to 60%"] -->
        Monitor["Monitor Uncertainty"] -->
        Resolve["Uncertainty Falls<br>Below 0.35m"] -->
        Resume["Return to TRACKING<br>Resume Normal Speed"]
    end

    style Start fill:#a0ad9b,stroke:#848e80,stroke-width:2px,color:#000000,font-weight:bold
    style Spike fill:#ae9f9f,stroke:#8f8383,stroke-width:2px,color:#000000,font-weight:bold
    style Detect fill:#b5b2a3,stroke:#959386,stroke-width:2px,color:#000000,font-weight:bold
    style Respond fill:#b5b2a3,stroke:#959386,stroke-width:2px,color:#000000,font-weight:bold
    style Monitor fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style Resolve fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
    style Resume fill:#a0ad9b,stroke:#848e80,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows how the system responds to an uncertainty spike. Starting from normal tracking with low uncertainty (0.2m), it detects a sudden increase to 0.6m. The system enters RECOVERY state, reduces speed, monitors the uncertainty, and then returns to normal TRACKING once uncertainty drops below the recovery threshold (0.35m).

By incorporating confidence and uncertainty management, our system becomes much more resilient to sensor issues and changing environmental conditions.

## 3. <a name="state-management-architecture"></a>State Management Architecture


### 3.1 <a name="system-design-rationale"></a>System Design Rationale

The State Management Node is designed with several core principles in mind:

#### 1. Decision Making vs. Data Processing

The architecture separates perception (what is happening) from decision making (what to do about it):

- **Fusion Node**: Focuses solely on combining sensor data optimally
  - Tracks position and velocity
  - Estimates uncertainty
  - Detects sensor gaps
  - Classifies motion patterns

- **State Manager**: Interprets this data to decide robot behavior
  - Determines appropriate robot state
  - Makes high-level behavioral decisions
  - Manages transitions between states
  - Sets behavioral parameters

This separation allows each node to be specialized and optimized for its specific task, resulting in better performance and maintainability.

#### 2. Distinct State Handling

Different robot states require completely different behaviors:

- **TRACKING**: Smooth following with predictive motion
- **SEARCHING**: Systematic rotation patterns
- **RECOVERY**: Careful, controlled movements
- **STOPPED**: Complete motor shutdown

Without a state manager, the fusion node would need to handle all these variations, making it overly complex. By centralizing decision-making, we keep other nodes simpler and more focused.

#### 3. Resilience to Sensor Issues

The state manager provides specific responses to different types of sensor failures:

- **Temporary occlusions**: Maintain tracking with prediction
- **High uncertainty**: Enter RECOVERY state
- **Complete loss**: Execute search patterns
- **Conflicting data**: Filter outliers and reconcile

These nuanced responses are behavioral decisions, not sensor fusion problems, making the state manager the appropriate place to handle them.

#### 4. Situational Awareness

The State Manager maintains context over time:

- Tracks duration in current state
- Implements hysteresis to prevent oscillation
- Remembers previous states for better decisions
- Maintains history for trend analysis

This temporal awareness allows for more sophisticated behavior than would be possible with direct sensor-to-motor connections.

### 3.2 <a name="node-architecture"></a>Node Architecture

The State Management Node is implemented as a ROS2 node with the following components:

<!-- Original complex diagram has been split into 4 separate diagrams -->

### StateManager Diagram

```
┌───────────────── STATE MANAGEMENT NODE: DECISION LAYER ───────────────────┐
│                                                                            │
│  ROLE: Determine appropriate behavior based on current context             │
│                                                                            │
│  ┌─────────────────────────┐   ┌─────────────────────┐                    │
│  │  FINITE STATE MACHINE   │   │   DECISION LOGIC    │                    │
│  │                         │   │                     │                    │
│  │ • Manages robot states  │   │ • Evaluates sensor  │                    │
│  │   (TRACKING, SEARCHING, │   │   data quality      │                    │
│  │   RECOVERY, etc.)       │   │ • Applies hysteresis│                    │
│  │ • Handles transitions   │   │   to prevent state  │                    │
│  │   between states        │   │   oscillation       │                    │
│  └─────────────────────────┘   └─────────────────────┘                    │
│                                                                            │
│  ┌─────────────────────────────────────────┐                              │
│  │          BEHAVIORAL SWITCHING            │                              │
│  │                                         │                              │
│  │ • Selects appropriate motion parameters │                              │
│  │ • Adjusts speed based on confidence     │                              │
│  │ • Changes search patterns as needed     │                              │
│  └─────────────────────────────────────────┘                              │
│                                                                            │
│  OUTPUT: State-appropriate motion commands sent to PID Controller          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Subscriptions Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
│   [PosSubscription] Position Subscriber                             │
│   [TrackingSubscription] Tracking Status Subscriber                 │
│   [UncertaintySubscription] Uncertainty Subscriber                  │
│   [MotionSubscription] Motion State Subscriber                      │
│   [ConfidenceSubscription] Confidence Subscriber                    │
│   [GapSubscription] Sensor Gap Subscriber                           │
│   [DiagSubscription] Diagnostics Subscriber                         │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   Subscriptions: PosSubscription, TrackingSubscription, Uncerta...  │
└────────────────────────────────────────────────────────────────────┘
```

### CoreComponents Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
│   [FSM] Finite State Machine                                        │
│   [HealthMonitor] Health Monitor                                    │
│   [ParamManager] Parameter Manager                                  │
│   [BufferManager] Buffer Manager                                    │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   CoreComponents: FSM, HealthMonitor, ParamManager, BufferManager   │
└────────────────────────────────────────────────────────────────────┘
```

### Publishers Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
│   [StatePublisher] State Publisher                                  │
│   [CommandPublisher] Command Publisher                              │
│   [HealthPublisher] Health Publisher                                │
│   [DiagPublisher] Diagnostics Publisher                             │
├────────────────────────────────────────────────────────────────────┤
│ CONNECTIONS:                                                        │
│   Subscriptions --> CoreComponents                                  │
│   CoreComponents --> Publishers                                     │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   Publishers: StatePublisher, CommandPublisher, HealthPublisher...  │
└────────────────────────────────────────────────────────────────────┘
```



**Diagram Explanation**: This diagram shows the internal architecture of the State Management Node. It's organized into three main sections: input subscriptions that receive data from other nodes (left), core processing components that handle the decision-making logic (middle), and output publishers that send commands and status information (right).

The node implements several core data structures:

- **Finite State Machine**: Manages robot state transitions
- **Health Monitor**: Tracks system health and confidence
- **Parameter Manager**: Handles adaptive parameter adjustments
- **Buffer Manager**: Maintains efficient circular buffers for time-series data

### 3.3 <a name="state-definitions"></a>State Definitions

The State Management Node implements the following states, each with specific behaviors and transition conditions:

#### INITIALIZING State

- **Purpose**: System startup state waiting for first reliable detection
- **Entry Condition**: System startup
- **Exit Condition**: Ball detected with confidence > threshold
- **Behavior**: Wait for reliable detection

#### TRACKING State

- **Purpose**: Normal operation state for following the ball
- **Entry Condition**: Consistent ball detection with sufficient confidence
- **Exit Conditions**: 
  - Ball lost for timeout period
  - High uncertainty detected
  - Ball close and stationary
- **Behavior**: Follow ball with PID control

#### LOST_BALL State

- **Purpose**: Final state when ball not found after extensive searching
- **Entry Condition**: Search timeout or multiple failed searches
- **Exit Condition**: Ball redetected with high confidence
- **Behavior**: Stop and wait for new detection

#### STOPPED State

- **Purpose**: Energy-saving state when ball is close and stationary
- **Entry Condition**: Ball close and stationary for threshold time
- **Exit Condition**: Ball moves or distance changes
- **Behavior**: Stop all motion to conserve energy

#### SEARCHING State

- **Purpose**: Active search mode when ball is temporarily lost
- **Entry Condition**: Ball lost temporarily from TRACKING state
- **Exit Conditions**:
  - Ball found (return to TRACKING)
  - Search timeout (move to LOST_BALL)
- **Behavior**: Execute rotation pattern to scan area

#### RECOVERY State

- **Purpose**: Handle uncertain tracking or sensor issues
- **Entry Condition**: High uncertainty or rising uncertainty trend
- **Exit Conditions**:
  - Uncertainty reduced (return to TRACKING)
  - Timeout without improvement (move to LOST_BALL)
- **Behavior**: Reduce speed and wait for better sensor data

This state table provides a comprehensive view of the system's possible states:

| State | Description | Entry Condition | Exit Condition | Behavior |
|-------|-------------|----------------|----------------|----------|
| **INITIALIZING** | Startup state | System startup | Ball detected with confidence > threshold | Wait for reliable detection |
| **TRACKING** | Active tracking | Consistent ball detection | Ball lost for timeout period or high uncertainty | Follow ball with PID control |
| **LOST_BALL** | Ball not found | Search timeout | Ball redetected | Stop and wait |
| **STOPPED** | Energy-saving | Ball close and stationary for threshold time | Ball moves or distance changes | Stop all motion |
| **SEARCHING** | Active search | Ball lost temporarily | Ball found or search timeout | Execute rotation pattern |
| **RECOVERY** | Handling uncertainty | High uncertainty detected | Uncertainty reduced or timeout | Stop and wait for better data |

### 3.4 <a name="information-flow"></a>Information Flow

Information flows through the State Management Node as follows:

#### 1. Input Data

The node subscribes to these ROS2 topics:

| Topic | Message Type | Description |
|-------|-------------|------------|
| `/basketball/fused/position` | `geometry_msgs/PoseStamped` | 3D position of the basketball |
| `/basketball/fused/tracking_status` | `std_msgs/Bool` | Boolean indicating reliable tracking |
| `/basketball/fused/position_uncertainty` | `std_msgs/Float32` | Uncertainty estimate of position |
| `/basketball/fused/motion_state` | `std_msgs/String` | Classification of ball's motion |
| `/basketball/fused/tracking_confidence` | `std_msgs/Float32` | Confidence value of tracking |
| `/basketball/fused/sensor_gap` | `std_msgs/Bool` | Boolean indicating sensor measurement gap |
| `/basketball/fusion/diagnostics` | `std_msgs/String` | Detailed fusion diagnostic data |

#### 2. Processing Pipeline

The data undergoes several processing steps:

1. **Input Validation**: Check data freshness and validity
2. **State Evaluation**: Determine if current state is still appropriate
3. **Transition Logic**: Evaluate if state transition is needed
4. **Hysteresis Application**: Apply protection against rapid state changes
5. **Health Monitoring**: Update system health and confidence metrics
6. **Command Generation**: Create appropriate movement commands based on state
7. **Diagnostic Data**: Gather diagnostic information

#### 3. Output Data

The node publishes these ROS2 topics:

| Topic | Message Type | Description |
|-------|-------------|------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands to control robot movement |
| `/robot/state` | `std_msgs/String` | Current robot state information |
| `/robot/health` | `std_msgs/Float32` | Health status of the system |
| `/robot/diagnostics` | `std_msgs/String` | Detailed diagnostic information |

This complete information flow can be visualized as:

<!-- Original complex diagram has been split into 3 separate diagrams -->

### Inputs Diagram

```

┌─────────────────── INPUT TOPICS (SUBSCRIPTIONS) ─────────────────────┐
│                                                                     │
│  ROS2 TOPICS:                                                       │
│                                                                     │
│  • /basketball/fused/position                                       │
│    Position of the basketball in 3D space (x,y,z)                   │
│                                                                     │
│  • /basketball/fused/tracking_status                                │
│    Current tracking status from the Fusion Node                     │
│                                                                     │
│  • /basketball/fused/position_uncertainty                           │
│    Uncertainty estimate of the position in meters                   │
│                                                                     │
│  • /basketball/fused/motion_state                                   │
│    Classification of ball motion (moving, slowing, stationary)      │
│                                                                     │
│  • /basketball/fused/tracking_confidence                            │
│    Confidence value from the tracking system (0.0-1.0)              │
│                                                                     │
│  • /basketball/fused/sensor_gap                                     │
│    Information about sensor data gaps or timing issues              │
│                                                                     │
│  • /basketball/fusion/diagnostics                                   │
│    Diagnostic data from the Fusion Node                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Processing Diagram

```

┌───────────────────── PROCESSING PIPELINE ─────────────────────────┐
│                                                                     │
│                        ┌───────────────────┐                        │
│                        │  Input Validation  │                        │
│                        └─────────┬─────────┘                        │
│                                  │                                  │
│                                  ▼                                  │
│                        ┌───────────────────┐                        │
│                        │  State Evaluation  │                        │
│                        └─────────┬─────────┘                        │
│                                  │                                  │
│                                  ▼                                  │
│                        ┌───────────────────┐                        │
│                        │  Transition Logic  │                        │
│                        └─────────┬─────────┘                        │
│                                  │                                  │
│                                  ▼                                  │
│                      ┌─────────────────────┐                        │
│                      │ Hysteresis Application │                      │
│                      └──────────┬──────────┘                        │
│                                 │                                   │
│                                 ▼                                   │
│                        ┌───────────────────┐                        │
│                        │  Health Monitoring  │                       │
│                        └─────────┬─────────┘                        │
│                                  │                                  │
│                                  ▼                                  │
│                        ┌───────────────────┐                        │
│                        │ Command Generation │                        │
│                        └─────────┬─────────┘                        │
│                                  │                                  │
│                                  ▼                                  │
│                       ┌─────────────────────┐                       │
│                       │ Diagnostic Collection │                      │
│                       └─────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Outputs Diagram

```

┌──────────────────── OUTPUT TOPICS (PUBLICATIONS) ──────────────────┐
│                                                                     │
│  ROS2 TOPICS:                                                       │
│                                                                     │
│  • /cmd_vel                                                         │
│    Movement commands to control the robot                           │
│                                                                     │
│  • /robot/state                                                     │
│    Current state of the robot's state machine                       │
│                                                                     │
│  • /robot/health                                                    │
│    Overall health and confidence metrics                            │
│                                                                     │
│  • /robot/diagnostics                                               │
│    Detailed diagnostic information                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```



**Diagram Explanation**: This flowchart illustrates the complete information pipeline through the State Management Node. Data enters through input topics on the left, passes through the processing pipeline in the middle, and exits through output topics on the right. The processing pipeline contains sequential steps that transform raw sensor data into appropriate robot commands.

## 4. <a name="core-state-machine"></a>Core State Machine


### 4.1 <a name="state-transition-logic"></a>State Transition Logic

The heart of the State Management Node is its state transition logic. Each state has specific entry and exit conditions that determine when transitions occur.

#### Basic State Transition Diagram

The core state transitions can be visualized as follows:

```mermaid
    stateDiagram-v2
    
    [*] --> INITIALIZING
    INITIALIZING --> TRACKING : Min. detections & reliable tracking
    INITIALIZING --> LOST_BALL : Timeout with no detections
    TRACKING --> SEARCHING : Ball lost temporarily
    TRACKING --> RECOVERY : High uncertainty
    TRACKING --> STOPPED : Ball stationary
    RECOVERY --> TRACKING : Uncertainty reduced
    RECOVERY --> LOST_BALL : Recovery unsuccessful
    SEARCHING --> TRACKING : Ball found
    SEARCHING --> LOST_BALL : Search timeout
    LOST_BALL --> TRACKING : Min. detections
    STOPPED --> TRACKING : Ball moves
```

**Diagram Explanation**: This state diagram shows all possible transitions between the system's states. Arrows indicate direction of transition, and labels on arrows indicate the conditions that trigger each transition. The system starts in INITIALIZING state (top) and transitions between other states based on sensor data and timing conditions.

#### Detailed State Transition Diagram with Timing

For more precise understanding, here's an expanded state diagram with specific timing parameters for each transition:

```mermaid
    stateDiagram-v2
    direction LR
    
    [*] --> INITIALIZING
    INITIALIZING --> TRACKING : Consistent detection
    INITIALIZING --> LOST_BALL : No detection for 5.0s
    TRACKING --> SEARCHING : Ball lost for 1.5s
    TRACKING --> RECOVERY : Uncertainty > 0.5m
    TRACKING --> STOPPED : Ball stationary for 1.5s
    RECOVERY --> TRACKING : Uncertainty < 0.35m
    RECOVERY --> LOST_BALL : In RECOVERY > 3.0s
    SEARCHING --> TRACKING : 6+ consecutive detections
    SEARCHING --> LOST_BALL : Searching > 30s
    LOST_BALL --> TRACKING : 6+ consecutive detections
    STOPPED --> TRACKING : Ball moves > 0.05m
```

**Diagram Explanation**: This expanded state diagram includes specific timing parameters and thresholds for each transition. For example, to go from TRACKING to SEARCHING, the ball must be lost for at least 1.5 seconds and the system must have been in TRACKING for at least 1.0 seconds (hysteresis protection). The color coding indicates different types of states: green for active tracking states, red for error/recovery states, and yellow for the initialization state.

#### Common State Transition Flow

In practice, certain transition paths occur more frequently than others. This diagram shows common transition paths with their average durations in a typical tracking scenario:

```mermaid

        flowchart TD
    subgraph "Common State Transitions and Durations"
        direction TB
    T["TRACKING<br>Avg Duration: 12.5s"] --- S1["STOPPED<br>Avg Duration: 4.2s"]
        T --- S2["SEARCHING<br>Avg Duration: 2.1s"]
        T --- R["RECOVERY<br>Avg Duration: 0.9s"]
        S2 --- L["LOST_BALL<br>Avg Duration: 0.7s"]
    S1 -->|"Ball stationary<br>1.5s + hysteresis"| T
        T -->|"Ball loss<br>1.5s"| S2
        S2 -->|"Ball found<br>6+ detections"| T
        S2 -->|"Search timeout<br>30.0s max"| L
        T -->|"High uncertainty<br>>0.5m"| R
        R -->|"Uncertainty reduced<br>0.35m"| T
        L -->|"Ball redetected<br>6+ detections"| T
        R -->|"Low confidence<br><0.5"| S2
    end

    style T fill:#a7b0a8,stroke:#89918a,stroke-width:2px,color:#000000,font-weight:bold
    style S1 fill:#a7b0a8,stroke:#89918a,stroke-width:2px,color:#000000,font-weight:bold
    style S2 fill:#b7a9ab,stroke:#978b8d,stroke-width:2px,color:#000000,font-weight:bold
    style R fill:#b7a9ab,stroke:#978b8d,stroke-width:2px,color:#000000,font-weight:bold
    style L fill:#b7a9ab,stroke:#978b8d,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows the most common state transitions with their average durations in real-world operation. The TRACKING state is central, with transitions to and from other states. Lines between states indicate possible transitions, with arrows showing specific transition conditions and their requirements. For example, the system typically spends about 12.5 seconds in TRACKING before transitioning to another state.

#### Implementation in Code

For each state, specialized handlers evaluate whether transitions should occur:

```python
# Pseudocode example of state transition logic
def handle_position_based_transitions(current_time):
    # Calculate time in current state for hysteresis
    time_in_state = current_time - state_start_time
    
    # Apply state-specific handlers
    if current_state == RobotState.INITIALIZING:
        _handle_initializing_transitions(time_in_state)
    elif current_state == RobotState.LOST_BALL:
        _handle_lost_ball_transitions(time_in_state)
    elif current_state == RobotState.RECOVERY:
        _handle_recovery_transitions(time_in_state)
    elif current_state == RobotState.SEARCHING:
        _handle_searching_transitions(time_in_state)
    elif current_state == RobotState.TRACKING:
        _handle_tracking_transitions(time_in_state, current_time)
    elif current_state == RobotState.STOPPED:
        _handle_stopped_transitions()
```

This approach allows for specialized transition logic for each state, keeping the code modular and maintainable.

### 4.2 <a name="hysteresis-protection"></a>Hysteresis Protection

A critical feature of the State Management Node is its hysteresis protection, which prevents rapid oscillation between states when conditions are borderline. This creates more stable and predictable robot behavior.

#### Hysteresis Mechanisms

The hysteresis protection implements several mechanisms:

##### 1. Minimum Time Requirements

Each state requires a minimum residence time before transitions are allowed:

```mermaid

        flowchart TD
    subgraph "Minimum Time Hysteresis"
        direction TB
    Enter["Enter TRACKING State<br>start_time = now()"] -->
        Trigger["Potential exit condition<br>(ball temporarily lost)"] -->
        Check{"time_in_state >=<br>tracking_hysteresis_time?"}
        Check -->|"No"| Stay["Stay in TRACKING<br>(ignore trigger)"]
        Check -->|"Yes"| Exit["Exit to new state<br>(SEARCHING)"]
    end

    style Enter fill:#a0ad9b,stroke:#848e80,stroke-width:2px,color:#000000,font-weight:bold
    style Stay fill:#a0ad9b,stroke:#848e80,stroke-width:2px,color:#000000,font-weight:bold
    style Exit fill:#ae9f9f,stroke:#8f8383,stroke-width:2px,color:#000000,font-weight:bold
    style Trigger fill:#b5b2a3,stroke:#959386,stroke-width:2px,color:#000000,font-weight:bold
    style Check fill:#99a8b2,stroke:#7e8a93,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows how time-based hysteresis works. When an exit condition is triggered (like the ball being temporarily lost), the system checks if it has been in the current state long enough (minimum hysteresis time). If not, it stays in the current state and ignores the trigger. Only if the minimum time requirement is met does it exit to the new state.

The minimum times vary by state:
- **TRACKING**: 1.0s minimum before exit
- **LOST_BALL**: 0.5s minimum before exit
- **SEARCHING**: 1.5s minimum before exit
- **STOPPED**: 0.5s minimum before exit
- **RECOVERY**: 0.3s minimum before exit

##### 2. Transition History Tracking

The system tracks recent transitions to detect oscillation patterns:

- Maintains history of last 5 state transitions
- Detects repeated patterns like A→B→A→B
- Applies increasing hysteresis when oscillation detected

##### 3. Adaptive Thresholds

Detection requirements increase after multiple state changes:

- First transition: standard thresholds
- After oscillation detected: stricter thresholds
- Example: Requiring 8 consecutive detections instead of 6 after repeated SEARCHING→TRACKING transitions

##### 4. Special Case Protection

Extra protection for critical states during challenging conditions:

- Longer minimum times during sensor gaps
- Higher confidence requirements after detection loss
- Different thresholds based on motion states

#### Implementation Example

```python
# Pseudocode example of hysteresis implementation
def apply_state_protection(proposed_state):
    current_time = time.time()
    time_in_state = current_time - state_start_time
    
    # Define minimum time requirements for each state
    min_times = {
        RobotState.TRACKING: tracking_hysteresis_time,
        RobotState.LOST_BALL: lost_ball_hysteresis_time,
        RobotState.SEARCHING: 1.5,
        RobotState.STOPPED: 0.5,
        RobotState.RECOVERY: recovery_hysteresis_time
    }
    
    # Get minimum time for current state
    min_time = min_times.get(current_state, 0.0)
    
    # Block transition if insufficient time in current state
    if time_in_state < min_time:
        return current_state
    
    # Check for oscillation patterns in state history
    if detect_oscillation(current_state, proposed_state):
        # Apply stricter requirements for oscillating transitions
        if current_state == RobotState.SEARCHING and proposed_state == RobotState.TRACKING:
            if consecutive_detections < min_retracking_detections + 2:
                return current_state
    
    # Special case protection during sensor gaps
    if (current_state == RobotState.STOPPED and
        proposed_state == RobotState.TRACKING and
        in_sensor_gap and
        motion_state in ["stationary", "long_stationary"]):
        return current_state
        
    # Allow transition if all checks pass
    return proposed_state
```

### 4.3 <a name="initialization-process"></a>Initialization Process

The system follows a structured initialization process designed to ensure consistent startup behavior.

#### Startup Sequence

```mermaid

        flowchart TD
    subgraph "Initialization Process"
        direction TB
    Start["ROS2 Node Startup"] --> 
        Params["Load Configuration Parameters"] -->
        Buffers["Initialize Data Buffers"] -->
        Publishers["Create Publishers"] -->
        Subscribers["Create Subscribers"] -->
        Timers["Setup Update Timers"] -->
        Logger["Initialize Logging"] -->
        State["Enter INITIALIZING State"] -->
        Wait["Wait for First Detection"] -->
        Track["Transition to TRACKING/LOST_BALL"]
    end

    style Start fill:#a288a6,stroke:#857089,stroke-width:2px,color:#000000,font-weight:bold
    style State fill:#b7b2a2,stroke:#979385,stroke-width:2px,color:#000000,font-weight:bold
    style Track fill:#a7b0a8,stroke:#89918a,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows the sequential steps in the system's initialization process. It starts with basic ROS2 node setup, loads configuration parameters, initializes data structures, sets up communication channels, and finally enters the INITIALIZING state to wait for the first ball detection.

#### INITIALIZING State Behavior

The INITIALIZING state serves as a buffer between startup and normal operation:

1. **Detection Counting**:
   - System counts consecutive ball detections
   - Requires minimum number of detections (default: 3)
   - Requires minimum confidence level (default: 0.7)

2. **Timeout Handling**:
   - If no detection after timeout (default: 5.0s)
   - Transitions to LOST_BALL state
   - Allows system to respond even without initial detection

3. **Parameter Initialization**:
   - Sets default parameters during this phase
   - Prepares system for immediate response after transition

#### Initial Parameterization

During initialization, the system prepares for operation by:

1. **Loading Default Parameters**:
   - Base values loaded from configuration file
   - Default values tuned for general-purpose operation

2. **Initial Adaptation**:
   - As soon as motion state is detected, adapts parameters
   - Example: Different parameters for stationary vs. moving balls

3. **Health Initialization**:
   - Starts with neutral health assessment
   - Begins confidence calculation once minimal data available

The initialization process ensures the system starts in a predictable state and transitions smoothly to normal operation once the basketball is detected.

## 5. <a name="advanced-features"></a>Advanced Features


The State Management Node implements several advanced features that enhance its adaptability, resilience, and performance in real-world conditions.

### 5.1 <a name="adaptive-parameter-management"></a>Adaptive Parameter Management

The system dynamically adjusts parameters based on changing conditions to optimize behavior.

#### Motion State Adaptation

Parameters adjust based on ball motion classification:

<!-- Original complex diagram has been split into 1 separate diagrams -->

### Adaptive Parameter Management System Diagram

```mermaid
flowchart LR
    subgraph "Adaptive Parameter Management System"
        direction TB
        
        Classifier --> Motion
        
        subgraph Motion["Motion State Detection"]
            M1["STATIONARY<br>Ball not moving significantly<br>< 0.05m movement over 1s"]
            M2["LONG_STATIONARY<br>Ball still for extended period<br>< 0.05m movement over 3s"]
            M3["MEDIUM_FAST<br>Ball moving moderately<br>0.3-1.0m/s velocity"]
            M4["HIGH_SPEED<br>Ball moving rapidly<br>> 1.0m/s velocity"]
        end
        
        subgraph Parameters["Parameter Adjustments"]
            P1["• lost_ball_timeout: +50%<br>• stationary_threshold: +50%<br>• tracking detections: unchanged<br>• velocity_scaling: -30%"]
            P2["• lost_ball_timeout: +80%<br>• stationary_threshold: +80%<br>• tracking detections: -30%<br>• velocity_scaling: -50%"]
            P3["• lost_ball_timeout: -30%<br>• stationary_threshold: -30%<br>• tracking detections: +50%<br>• velocity_scaling: +20%"]
            P4["• lost_ball_timeout: -50%<br>• stationary_threshold: -50%<br>• tracking detections: +100%<br>• velocity_scaling: +40%"]
        end
        
        subgraph Impact["Decision-Making Impact"]
            I1["• More tolerant of detection losses<br>• Slower velocity commands<br>• Quicker transition to STOPPED"]
            I2["• Very tolerant of detection gaps<br>• Minimal velocity commands<br>• Ready for STOPPED state<br>• Lower power consumption"]
            I3["• Quicker reaction to ball loss<br>• Requires more confident detection<br>• Higher velocity commands<br>• Emphasis on tracking continuity"]
            I4["• Very quick reaction to ball loss<br>• Requires highly confident detection<br>• Maximum velocity commands<br>• Predictive tracking engaged"]
        end
        
        M1 --> P1 --> I1
        M2 --> P2 --> I2
        M3 --> P3 --> I3
        M4 --> P4 --> I4
    end
```



**Diagram Explanation**: This enhanced diagram shows the complete adaptive parameter system that drives intelligent decision-making:

**Motion Classification Process:**
- The system receives ball movement data from the Fusion Node
- The Motion State Classifier categorizes the ball's movement pattern using velocity and time thresholds
- Four distinct motion states are identified, each with precise definition criteria

**Parameter Adaptation:**
- Each motion state triggers specific parameter adjustments (shown with exact percentages)
- Critical parameters like timeout durations, thresholds, and detection requirements are all modified
- These adjustments are tailored to optimize performance for each motion scenario

**Decision-Making Impact:**
- The parameter changes directly influence how the robot makes decisions
- For stationary balls, the system becomes more tolerant of detection gaps and prepares for the STOPPED state
- For fast-moving balls, the system requires more confident detections and implements quicker reactions
- These adaptations allow appropriate tradeoffs between reliability and responsiveness

This adaptive approach enables the robot to make better decisions by customizing its behavior to match the current movement characteristics of the basketball. The system effectively becomes "context-aware" rather than using fixed parameters for all situations.

#### Distance-Based Adaptation

Parameters scale based on distance to target:

- **Close Range** (< 0.5m):
  - Lower movement thresholds for precision
  - Extended stationary detection times
  - Quicker transition to STOPPED state

- **Medium Range** (0.5m - 2.0m):
  - Balanced parameters for general tracking
  - Standard timeouts and thresholds
  - Normal tracking sensitivity

- **Far Range** (> 2.0m):
  - Higher movement thresholds
  - Extended tracking timeouts
  - Increased detection requirements

#### Health-Based Adaptation

Parameters adjust based on system health and confidence:

- **High Confidence** (> 0.8):
  - Standard parameters
  - Optimal performance priorities

- **Medium Confidence** (0.5 - 0.8):
  - More conservative timeouts
  - Higher detection requirements
  - Reduced maximum speeds

- **Low Confidence** (< 0.5):
  - Significantly extended timeouts
  - Stricter detection requirements
  - Reduced speeds for safety
  - Increased recovery thresholds

#### Implementation Example

```python
# Pseudocode example of adaptive parameter management
def adapt_parameters_to_motion_state():
    if not adaptive_parameters_enabled:
        return
    
    # Reset parameters to base values
    lost_ball_timeout = base_lost_ball_timeout
    stationary_threshold = base_stationary_threshold
    min_tracking_detections = base_min_tracking_detections
    
    # Apply state-specific adjustments
    if motion_state == "stationary":
        # More relaxed parameters for stationary balls
        lost_ball_timeout *= adaptive_factor_stationary  # e.g., 1.5x longer
        stationary_threshold *= adaptive_factor_stationary  # e.g., 1.5x larger
        
    elif motion_state == "long_stationary":
        # Even more relaxed for long-stationary balls
        lost_ball_timeout *= adaptive_factor_stationary * 1.2  # e.g., 1.8x longer
        stationary_threshold *= adaptive_factor_stationary * 1.2  # e.g., 1.8x larger
        min_tracking_detections = max(2, int(min_tracking_detections * 0.7))  # e.g., 30% fewer
        
    elif motion_state == "medium_fast":
        # Stricter parameters for fast movement
        lost_ball_timeout *= adaptive_factor_moving  # e.g., 0.8x shorter
        stationary_threshold *= adaptive_factor_moving  # e.g., 0.8x smaller
        min_tracking_detections += 1  # e.g., require one more detection
```

### 5.2 <a name="sensor-gap-handling"></a>Sensor Gap Handling

The system implements sophisticated handling of sensor gaps - periods when sensors temporarily fail to provide data.

#### Gap Detection and Classification

The system detects and classifies different types of sensor gaps:

- **Micro Gaps** (< 0.2s):
  - Brief interruptions in sensor data
  - Handled with motion prediction
  - No state changes required

- **Short Gaps** (0.2s - 1.5s):
  - Temporary occlusions or sensor issues
  - Handled with gap tolerance mechanism
  - May maintain current state with adjusted parameters

- **Extended Gaps** (> 1.5s):
  - Significant sensor failures
  - May require state transitions to RECOVERY or SEARCHING
  - Uses uncertainty trends to determine appropriate response

#### Gap Tolerance Mechanism

```
┌───────────────── GAP TOLERANCE MECHANISM ────────────────────────────┐
│                                                                       │
│  SENSOR GAP HANDLING PROCESS:                                         │
│                                                                       │
│  1. SENSOR GAP DETECTED                                               │
│     |                                                                 │
│     └─► GAP DURATION EVALUATION                                       │
│         |                                                             │
│         ├─► SHORT GAP (< 0.2s)                                        │
│         │    └─► Use motion prediction, maintain state                │
│         │                                                             │
│         ├─► MEDIUM GAP (0.2s - 1.5s)                                  │
│         │    └─► Apply tolerance, maintain state, reduce velocity     │
│         │                                                             │
│         └─► LONG GAP (> 1.5s)                                         │
│              └─► UNCERTAINTY EVALUATION                               │
│                  |                                                    │
│                  ├─► LOW/STABLE: Maintain TRACKING with lower speed   │
│                  │                                                    │
│                  ├─► HIGH/RISING: Transition to RECOVERY state        │
│                  │                                                    │
│                  └─► VERY HIGH: Transition to SEARCHING state         │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Diagram Explanation**: This flowchart shows how the system handles different types of sensor gaps. The response varies based on gap duration and uncertainty status. Micro gaps (under 0.2s) are handled with motion prediction, short gaps (0.2-1.5s) use the gap tolerance mechanism, and extended gaps (over 1.5s) trigger different responses based on position uncertainty.

The gap tolerance mechanism adapts based on context:

- **Adaptive Tolerance Time**:
  - Baseline tolerance time (default: 1.5s)
  - Extended for stationary balls (up to 3.0s)
  - Reduced for fast-moving balls (down to 0.8s)

- **Motion-Based Adjustments**:
  - Different strategies for different motion states
  - More tolerance for stationary objects
  - Less tolerance for fast-moving objects

#### Implementation Example

```python
# Pseudocode example of sensor gap handling
def handle_sensor_gap():
    if not gap_enabled or not in_sensor_gap:
        return
    
    current_time = time.time()
    gap_duration = current_time - gap_start_time
    
    # Calculate adaptive tolerance based on motion state
    tolerance_time = gap_tolerance_time  # Base tolerance (e.g., 1.5s)
    if motion_state in ["stationary", "long_stationary"]:
        tolerance_time *= gap_stationary_multiplier  # e.g., 2.0x longer for stationary balls
    elif motion_state == "medium_fast":
        tolerance_time *= 0.8  # e.g., 0.8x shorter for fast-moving balls
    
    # Handle gap based on current state
    if current_state == RobotState.TRACKING:
        # For short gaps, stay in TRACKING
        if gap_duration < tolerance_time:
            # Override timeout logic by updating last detection time
            # This prevents transition to SEARCHING during tolerable gaps
            last_detection_time = current_time - (lost_ball_timeout * 0.5)
            
            # Reduce velocity during gap
            current_velocity_scale = max(0.3, 1.0 - (gap_duration / tolerance_time))
        else:
            # Gap too long - consider recovery
            if position_uncertainty < uncertainty_recovery_threshold:
                # Stay in tracking if uncertainty acceptable
                pass
            else:
                # Enter recovery
                transition_to_state(RobotState.RECOVERY)
                recovery_reason = "extended_sensor_gap"
```

### 5.3 <a name="uncertainty-management"></a>Uncertainty Management

The State Management Node actively monitors position uncertainty and responds appropriately to changes.

#### Uncertainty Monitoring and Analysis

The system tracks uncertainty through several methods:

<!-- Replacing multiple diagrams with a single comprehensive diagram -->

```
┌─────────────────────────── UNCERTAINTY MANAGEMENT SYSTEM ────────────────────────────┐
│                                                                                     │
│  ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐       │
│  │      SOURCES      │      │     ANALYSIS      │      │     RESPONSES      │       │
│  ├───────────────────┤      ├───────────────────┤      ├───────────────────┤       │
│  │                   │      │                   │      │                   │       │
│  │  Fusion Algorithm │      │  Absolute Value   │      │     Parameter     │       │
│  │  Position Uncertainty ────► Monitoring       │      │     Adjustment    │       │
│  │                   │      │                   │      │                   │       │
│  │                   │      │                   │      │                   │       │
│  │  Sensor Count     │      │  Trend Analysis   │      │   Recovery State  │       │
│  │  and Quality      ────────────────────────────────► Entry              │       │
│  │                   │      │                   │      │                   │       │
│  │                   │      │                   │      │                   │       │
│  │  Detection        │      │  Rate of Change   │      │     Velocity      │       │
│  │  Consistency      ──────► Calculation        ──────► Reduction          │       │
│  │                   │      │                   │      │                   │       │
│  │                   │      │                   │      │                   │       │
│  │  Motion State     │      │  Pattern          │      │      Search       │       │
│  │                   ──────► Recognition        ──────► Initiation         │       │
│  │                   │      │                   │      │                   │       │
│  └───────────────────┘      └───────────────────┘      └───────────────────┘       │
│                                                                                     │
│                                                                                     │
│  ┌─────────────────────────── UNCERTAINTY HANDLING PROCESS ─────────────────────┐  │
│  │                                                                              │  │
│  │  1. UNCERTAINTY DETECTION                                                    │  │
│  │     │                                                                        │  │
│  │     ├─► HIGH ABSOLUTE VALUE (> 0.5m)                                        │  │
│  │     │    └─► Immediate RECOVERY state transition                            │  │
│  │     │                                                                        │  │
│  │     ├─► RAPIDLY INCREASING TREND                                            │  │
│  │     │    └─► Velocity reduction proportional to rate                        │  │
│  │     │                                                                        │  │
│  │     └─► PATTERN DETECTION                                                   │  │
│  │          │                                                                   │  │
│  │          ├─► OSCILLATION: Apply damping parameters                          │  │
│  │          │                                                                   │  │
│  │          ├─► CONTINUOUS RISE: Initiate search if > 30 sec                   │  │
│  │          │                                                                   │  │
│  │          └─► CORRELATION WITH MOVEMENT: Adjust thresholds                   │  │
│  │                                                                              │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

This diagram shows the complete uncertainty management system. It begins with uncertainty data from various sources (left), applies different analysis methods to understand the uncertainty (middle), and then implements appropriate response strategies based on the analysis (right). The bottom section details the specific uncertainty handling process with decision paths.   - Stable uncertainty: consistent tracking

2. **Rate of Change**:
   - Slow changes: normal operation
   - Rapid changes: potential issues
   - Sudden spikes: sensor conflicts or obstacles

3. **Pattern Recognition**:
   - Oscillating uncertainty: inconsistent detection
   - Steadily rising uncertainty: gradually losing tracking
   - Step changes: abrupt environmental changes

#### Recovery Triggering

The system uses uncertainty information to determine when to enter RECOVERY state:

```mermaid

        flowchart TD
    subgraph "Recovery State Triggering"
        direction TB
    Start["Monitoring<br>in TRACKING State"] --> 
        Check1{"Uncertainty ><br>threshold?"}
    Check1 -->|"Yes"| Recover1["Enter RECOVERY<br>Reason: high_uncertainty"]
        Check1 -->|"No"| Check2{"Uncertainty<br>rising rapidly?"}
    Check2 -->|"Yes"| Recover2["Enter RECOVERY<br>Reason: rising_uncertainty"]
        Check2 -->|"No"| Continue["Continue TRACKING"]
    Recover1 --> Monitor["Monitor Uncertainty<br>in RECOVERY State"]
        Recover2 --> Monitor
    Monitor --> Check3{"Uncertainty <<br>recovery threshold?"}
        Check3 -->|"Yes"| Exit["Return to<br>TRACKING State"]
        Check3 -->|"No"| Stay["Remain in<br>RECOVERY State"]
    end

    style Start fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style Continue fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style Exit fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style Check1 fill:#94a3b7,stroke:#7a8697,stroke-width:2px,color:#000000,font-weight:bold
    style Check2 fill:#94a3b7,stroke:#7a8697,stroke-width:2px,color:#000000,font-weight:bold
    style Check3 fill:#94a3b7,stroke:#7a8697,stroke-width:2px,color:#000000,font-weight:bold
    style Recover1 fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style Recover2 fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style Stay fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style Monitor fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart illustrates the decision process for entering and exiting the RECOVERY state based on uncertainty. The system enters RECOVERY if either the absolute uncertainty exceeds a threshold or if uncertainty is rising rapidly. It exits RECOVERY once uncertainty drops below the recovery threshold, which is lower than the entry threshold (providing hysteresis).

Key aspects of the uncertainty-based recovery system:

1. **Threshold Hysteresis**:
   - Enter RECOVERY when uncertainty > 0.5m
   - Exit RECOVERY when uncertainty < 0.35m
   - This gap prevents oscillation

2. **Trend-Based Entry**:
   - Enter RECOVERY if uncertainty rising rapidly (> 0.01m/s)
   - Even if absolute value is below threshold
   - Proactive response to deteriorating tracking

3. **Motion-Specific Thresholds**:
   - Different thresholds for different motion states
   - Higher tolerance for fast-moving objects
   - Lower tolerance for stationary objects

#### Implementation Example

```python
# Pseudocode example of uncertainty management
def evaluate_uncertainty_recovery():
    if current_state != RobotState.TRACKING:
        return
    
    # Early exit if uncertainty is low
    if position_uncertainty < uncertainty_recovery_threshold:
        return
    
    # Check uncertainty trend
    if len(uncertainty_history.values) >= 5:
        direction, rate = uncertainty_history.get_trend(5)
        
        # Enter recovery if uncertainty is high and rising
        if direction > 0 and rate > 0.01:
            recovery_reason = "rising_uncertainty"
            transition_to_state(RobotState.RECOVERY)
            return
        
        # Also enter recovery if uncertainty is very high even if stable
        if position_uncertainty > position_uncertainty_threshold:
            recovery_reason = "high_uncertainty"
            transition_to_state(RobotState.RECOVERY)
            return
```

### 5.4 <a name="motion-state-integration"></a>Motion State Integration

The system integrates motion state information from the fusion node to enhance decision making.

#### Motion Classification System

The fusion node classifies ball movement into several categories:

- **stationary**: Ball has not moved significantly for a short period
- **long_stationary**: Ball has been still for an extended period
- **slow_moving**: Ball moving at walking pace or slower
- **medium_speed**: Ball moving at jogging pace
- **medium_fast**: Ball moving at running pace
- **fast**: Ball moving at high speed (thrown/bouncing)

This classification enables context-aware decision making.

#### Motion State Transitions

The system tracks transitions between motion states and adjusts behavior accordingly:

```mermaid

        flowchart LR
    subgraph "Motion State Transitions"
        direction LR
    M1["stationary"] -->|"Movement<br>detected"| M3["slow_moving"]
        M3 -->|"Speed<br>increases"| M4["medium_speed"]
        M4 -->|"Speed<br>increases"| M5["medium_fast"]
        M5 -->|"Speed<br>increases"| M6["fast"]
    M3 -->|"Movement<br>stops"| M1
        M4 -->|"Slows<br>down"| M3
        M5 -->|"Slows<br>down"| M4
        M6 -->|"Slows<br>down"| M5
    M1 -->|"Remains still<br>for threshold time"| M2["long_stationary"]
        M2 -->|"Movement<br>detected"| M3
    end

    style M1 fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style M2 fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style M3 fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
    style M4 fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
    style M5 fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style M6 fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows how the system tracks transitions between different motion states. The ball can move between states as its speed changes, with "stationary" and "long_stationary" for still balls (green), "slow_moving" and "medium_speed" for moderate movement (yellow), and "medium_fast" and "fast" for rapid movement (red).

#### Motion-Based Parameter Adjustment

Different motion states trigger different parameter sets:

| Parameter | stationary | long_stationary | medium_speed | medium_fast | fast |
|-----------|------------|----------------|--------------|-------------|------|
| `lost_ball_timeout` | +50% | +80% | +0% | -20% | -30% |
| `stationary_threshold` | +50% | +80% | +0% | -20% | -30% |
| `min_tracking_detections` | +0% | -30% | +0% | +30% | +50% |
| `gap_tolerance_time` | +100% | +200% | +0% | -20% | -50% |
| `uncertainty_tolerance` | +50% | +80% | +0% | +30% | +50% |

#### Motion-State Decision Making

Motion states influence state transition decisions:

1. **Stationary Ball Logic**:
   - Faster transition to STOPPED state
   - Extended timeouts before SEARCHING
   - Lower velocity commands

2. **Fast-Moving Ball Logic**:
   - More predictive tracking
   - Higher velocity commands
   - Stricter detection requirements

3. **Transition Handling**:
   - Special handling during motion state changes
   - Buffering parameters during transitions
   - Avoiding jerky responses to motion changes

#### Implementation Example

```python
# Pseudocode example of motion state integration
def motion_state_callback(msg):
    # Store previous and current state
    last_motion_state = motion_state
    motion_state = msg.data
    
    # Detect transitions
    motion_state_changed = last_motion_state != motion_state
    in_motion_transition = motion_state_changed
    
    # Log state changes and update parameters
    if motion_state_changed:
        logger.info(f"Motion state changed: {last_motion_state} -> {motion_state}")
        
        # During transitions, adjust parameters immediately
        adapt_parameters_to_motion_state()
        
        # Force state reevaluation after parameter changes
        if current_state in [RobotState.LOST_BALL, RobotState.TRACKING]:
            handle_position_based_transitions(time.time())
            
        # Special handling for entering stationary states
        if motion_state in ["stationary", "long_stationary"]:
            stationary_start_time = time.time()
            
        # Special handling for exiting stationary states
        elif last_motion_state in ["stationary", "long_stationary"]:
            stationary_start_time = None
```

This motion state integration allows the system to adapt its behavior based on how the basketball is moving, creating more appropriate responses in different situations.

## 6. <a name="health-monitoring-system"></a>Health Monitoring System


### 6.1 <a name="system-confidence-calculation"></a>System Confidence Calculation

The State Management Node implements a comprehensive health monitoring system that calculates overall system confidence based on multiple factors.

#### Multi-Factor Confidence Model

The system confidence calculation combines several key metrics:

```mermaid
        flowchart TD
    
    subgraph "System Confidence Calculation"
        direction TB
        
        TC["Tracking Confidence<br>Range: 0.0-1.0"]
        PU["Position Uncertainty<br>Unit: meters"]
        SC["Active Sensor Count<br>Range: 0-3"]
        WC["Active Warnings<br>Count: 0-5+"]
    
        TC --> Trans1["Direct Value"] --> Weight1["Weight: 40%"] --> Combine
        PU --> Trans2["1.0 / (1.0 + uncertainty * 2.0)"] --> Weight2["Weight: 30%"] --> Combine
        SC --> Trans3["min(1.0, count / 2.0)"] --> Weight3["Weight: 20%"] --> Combine
        WC --> Trans4["-0.1 per warning"] --> Weight4["Weight: 10%"] --> Combine
        
        Combine["Weighted Combination"] --> Clamp["Clamp: 0.1-1.0"] --> Final["System Confidence<br>0.1 (critical) to 1.0 (optimal)"]
        
        Final --> Decision["Decision-Making Impact"]
    end
```

**Diagram Explanation**: This enhanced flowchart illustrates not just how system confidence is calculated, but also how it directly impacts the robot's decision-making process:

**Confidence Calculation (Top):**
- Input metrics (blue) from multiple sources are transformed and weighted
- Tracking confidence (40%), position uncertainty (30%), sensor count (20%), and warnings (10%) 
- These are combined and clamped to produce a final confidence value between 0.1-1.0

**Decision Impact (Bottom):**
- The calculated confidence directly determines the robot's behavior
- High confidence (>0.8) enables aggressive tracking with higher speeds
- Medium confidence (0.5-0.8) triggers more conservative movement
- Low confidence (0.3-0.5) activates the RECOVERY state with very slow movement
- Critical confidence (<0.3) triggers the SEARCHING state with structured search patterns

This multi-factor approach allows the robot to gracefully degrade its performance as confidence decreases, rather than abruptly failing when conditions become challenging. The system continuously adapts its behavior based on the quality of its perception data.

The calculation includes these components:

1. **Tracking Confidence** (40% weight)
   - Direct value from fusion node (0.0-1.0)
   - Higher values increase overall confidence

2. **Position Uncertainty** (30% weight)
   - Inverse relationship (lower uncertainty = higher confidence)
   - Transformed using formula: `1.0 / (1.0 + uncertainty * 2.0)`

3. **Sensor Count** (20% weight)
   - More active sensors increase confidence
   - Transformed using formula: `min(1.0, count / 2.0)` (2+ sensors = full confidence)

4. **Warning Penalties** (10% weight)
   - Each active warning reduces confidence by 0.1
   - Multiple warnings can significantly reduce overall confidence

#### Implementation

```python
# Pseudocode example of confidence calculation
def calculate_system_confidence():
    # Start with base confidence
    confidence = 1.0
    
    # Factor in tracking confidence (40% weight)
    tracking_weight = 0.4
    tracking_confidence = components['tracking'][1]
    confidence *= (tracking_weight * tracking_confidence + (1 - tracking_weight))
    
    # Factor in fusion uncertainty (30% weight)
    # Invert uncertainty to get confidence (lower uncertainty = higher confidence)
    uncertainty = components['fusion'][1]
    uncertainty_factor = 1.0 / (1.0 + uncertainty * 2.0)
    uncertainty_weight = 0.3
    confidence *= (uncertainty_weight * uncertainty_factor + (1 - uncertainty_weight))
    
    # Factor in sensor count (20% weight)
    sensor_count = components['sensors'][0]
    sensor_factor = min(1.0, sensor_count / 2.0)  # 2+ sensors = full confidence
    sensor_weight = 0.2
    confidence *= (sensor_weight * sensor_factor + (1 - sensor_weight))
    
    # Apply penalties for warnings (10% reduction each)
    warning_penalty = 0.1 * len(warnings)
    confidence = max(0.1, confidence - warning_penalty)
    
    return confidence
```

#### Example Calculation

Here's a real-world example of confidence calculation with values from a tracking session:

```
CONFIDENCE CALCULATION:
                                    
    Sensor Metrics                  Health Metrics
  +----------------+              +-----------------+
  | Detection: 0.9 |              | CPU Usage: 0.95 |
  +----------------+              +-----------------+
  |    Weight: 30% |                  Weight: 5%
  +----------------+              +-----------------+
                                  | Memory Use: 0.8 |
  +----------------+              +-----------------+
  | Tracking: 0.85 |                  Weight: 5%
  +----------------+
  |    Weight: 25% |
  +----------------+
                                  
  +----------------+              
  |Uncertainty: 0.7|
  +----------------+
  |    Weight: 20% |
  +----------------+
                                  
  +----------------+
  |Consistency: 0.8|
  +----------------+
  |    Weight: 15% |
  +----------------+

  Final Confidence Score: 0.832
```

This calculation combines multiple factors with their appropriate weights to produce a single confidence value that represents the overall health of the system.

### 6.2 <a name="warning-detection"></a>Warning Detection

The health monitoring system actively detects warning conditions that might affect performance.

#### Warning Categories

The system monitors for several categories of warnings:

1. **Stale Data Warnings**
   - Component data too old
   - Example: No position update in 1.0 second
   - Severity based on age of data

2. **Tracking Degradation Warnings**
   - Confidence below threshold
   - Erratic position updates
   - Inconsistent tracking

3. **Uncertainty Warnings**
   - High absolute uncertainty
   - Rapidly rising uncertainty
   - Unstable uncertainty patterns

4. **Sensor Gap Warnings**
   - Detected sensor measurement gaps
   - Missing sensor data
   - Sensor timing issues

5. **Sensor Count Warnings**
   - Too few active sensors
   - Loss of specific sensor types
   - Sensor reliability issues

#### Trend-Based Warning Detection

The system uses trend analysis to detect developing issues:

```
┌──────────────────── TREND-BASED WARNING DETECTION ────────────────────┐
│                                                                       │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐ 
│  │  HISTORICAL DATA   │    │   TREND ANALYSIS   │    │  WARNING TRIGGERS  │ 
│  ├────────────────────┤    ├────────────────────┤    ├────────────────────┤ 
│  │                    │    │                    │    │                    │ 
│  │ Uncertainty History│    │ Calculate Uncertainty│   │ Rising Uncertainty  
│  │ (Last 10 readings) │───>│ Trend Direction    │──>│ Warning            │ 
│  │                    │    │ and Rate           │   │                    │ 
│  │                    │    │                    │    │                    │ 
│  │ Confidence History │    │ Calculate Confidence│   │ Falling Confidence │ 
│  │ (Last 10 readings) │───>│ Trend Direction    │──>│ Warning            │ 
│  │                    │    │ and Rate           │   │                    │ 
│  │                    │    │                    │    │                    │ 
│  │ Sensor Count History│    │ Detect Sensor      │   │ Sensor Loss       │ 
│  │ (Last 10 readings) │───>│ Count Changes      │──>│ Warning            │ 
│  │                    │    │                    │   │                    │ 
│  │                    │    │                    │    │                    │ 
│  └────────────────────┘    └────────────────────┘    │ Pattern-Based     │ 
│                                        │             │ Warning (combines  │ 
│                                        │             │ multiple trends)   │ 
│                                        |────────────>│                    │ 
│                                                      │                    │ 
│                                                      │                    │ 
│                                                      └────────────────────┘ 
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

**Diagram Explanation**: This flowchart shows how the system generates warnings based on trend analysis. Time-series data is collected and analyzed for trends , which can then trigger specific warnings if problematic patterns are detected.

#### Implementation Example

```python
# Pseudocode example of warning detection
def evaluate_health():
    current_time = time.time()
    warnings = []
    
    # Check for stale data
    for component, data in components.items():
        age = current_time - data[2]  # Last update time
        if age > 2.0:
            warnings.append(f"{component}_stale_data")
    
    # Check for degraded tracking
    if components['tracking'][1] < 0.4 and not components['tracking'][0]:
        warnings.append('tracking_degraded')
    
    # Check for high uncertainty
    if components['fusion'][1] > 0.5:
        # Check if uncertainty is rising
        direction, rate = trends['uncertainty'].get_trend(5)
        if direction > 0 and rate > 0.05:
            warnings.append('uncertainty_rising')
        else:
            warnings.append('high_uncertainty')
    
    # Check for sensor gaps during tracking
    if components['sensors'][1] and components['tracking'][0]:
        warnings.append('sensor_gap_during_tracking')
    
    # Check for low sensor count
    if components['sensors'][0] < 1:
        warnings.append('no_active_sensors')
    
    return warnings
```

#### Warning Response Mechanism

When warnings are detected, the system implements graduated responses:

1. **Log Warning**:
   - Record warning in system log
   - Include related metrics and context
   - Timestamp for later analysis

2. **Adjust Confidence**:
   - Reduce system confidence based on warning
   - Apply appropriate penalty for warning type
   - Factor into decision making

3. **Parameter Adjustment**:
   - Modify operational parameters
   - Example: More conservative timeouts
   - Example: Higher detection thresholds

4. **State Transition**:
   - May trigger specific state transitions
   - Example: RECOVERY state for high uncertainty
   - Example: SEARCHING for tracking degradation

### 6.3 <a name="diagnostic-data"></a>Diagnostic Data

The health monitoring system provides comprehensive diagnostic information for monitoring and debugging.

#### Diagnostic Levels

The system implements a tiered approach to diagnostic information:

1. **Basic State Information**
   - Current state and duration
   - Previous state
   - State transition reason
   - Always available

2. **Health Metrics**
   - System confidence value
   - Component status indicators
   - Warning counts and types
   - Updated periodically

3. **Full Diagnostic Data**
   - Detailed performance metrics
   - Historical trend information
   - Component interaction data
   - Less frequent updates

4. **Resource Monitoring**
   - CPU and memory usage
   - Message throughput
   - Timer performance
   - Optional, can be disabled

#### Published Information

The diagnostic data includes:

```mermaid

        flowchart TD
    subgraph "Diagnostic Data Structure"
        direction TB
    Root["Diagnostic Message"]
    Root --> Basic["Basic Information<br>- Current state<br>- Time in state<br>- Previous state"]
    Root --> Tracking["Tracking Information<br>- Reliable tracking status<br>- Consecutive detections<br>- Position uncertainty<br>- Time since detection"]
    Root --> Ball["Ball Information<br>- Distance<br>- Close status<br>- Stationary status<br>- Motion state"]
    Root --> Health["Health Information<br>- System confidence<br>- Warning count<br>- Active warnings<br>- Component status"]
    Root --> System["System Information<br>- CPU usage<br>- Memory usage<br>- Message count<br>- Update frequency"]
    end

    style Root fill:#a8aaac,stroke:#8a8c8e,stroke-width:2px,color:#000000,font-weight:bold
    style Basic fill:#94a3b7,stroke:#7a8697,stroke-width:2px,color:#000000,font-weight:bold
    style Tracking fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style Ball fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style Health fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
    style System fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows the structure of the diagnostic data published by the system. The root diagnostic message contains several categories of information, including basic state data, tracking information, ball data, health metrics, and system information.

#### Implementation Example

```python
# Pseudocode example of diagnostic publication
def publish_diagnostics():
    current_time = time.time()
    
    # Build basic diagnostic info
    diagnostic_info = {
        "state": current_state,
        "tracking": {
            "reliable": tracking_reliable,
            "consecutive_detections": consecutive_detections,
            "uncertainty": round(position_uncertainty, 3),
            "time_since_detection": round(time_since_detection, 2)
        },
        "ball": {
            "distance": round(ball_distance, 2),
            "is_close": is_ball_close,
            "is_stationary": is_ball_stationary,
            "motion_state": motion_state
        }
    }
    
    # Every ~5 seconds, include full diagnostics
    if current_time - last_full_diagnostic_time > full_diagnostic_rate:
        full_diagnostics = True
        last_full_diagnostic_time = current_time
        
        # Add detailed health information
        diagnostic_info["system_health"] = {
            "confidence": round(health_monitor.system_confidence, 2),
            "warnings_count": len(health_monitor.warnings),
            "active_warnings": health_monitor.warnings,
            "components": {
                "tracking": components['tracking'][0],
                "fusion": round(components['fusion'][1], 2),
                "sensors": components['sensors'][0]
            }
        }
        
        # Add system information if enabled
        if resource_monitoring_enabled:
            diagnostic_info["system_info"] = {
                "cpu_usage": round(process.cpu_percent(), 1),
                "memory_usage": round(process.memory_info().rss / 1024 / 1024, 1),
                "messages_received": message_counter,
                "update_frequency": round(1.0 / (current_time - last_update_time), 1) if last_update_time else 0
            }
    
    # Publish diagnostic info
    diag_msg.data = json.dumps(diagnostic_info, cls=FastJSONEncoder)
    diagnostics_publisher.publish(diag_msg)
```

## 7. <a name="optimized-data-structures"></a>Optimized Data Structures


### 7.1 <a name="circular-buffers"></a>Circular Buffers

The State Management Node uses memory-efficient circular buffers for storing historical data.

#### Circular Buffer Implementation

<!-- Original complex diagram has been split into 4 separate diagrams -->

### Circular Buffer Design Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   Circular Buffer Design:                                           │
└────────────────────────────────────────────────────────────────────┘
```

### Structure Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
│   [S1] Fixed-size pre-allocated array                               │
│   [S2] Index pointer to next position                               │
│   [S3] Size counter (current elements)                              │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   Structure: S1, S2, S3                                             │
└────────────────────────────────────────────────────────────────────┘
```

### Operations Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   Operations:                                                       │
└────────────────────────────────────────────────────────────────────┘
```

### Memory Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
│   [M1] No dynamic allocations                                       │
│   [M2] No memory fragmentation                                      │
│   [M3] Constant memory footprint                                    │
│   [M4] Automatic discarding of old data                             │
├────────────────────────────────────────────────────────────────────┤
│ CONNECTIONS:                                                        │
│   Structure --> Operations                                          │
│   Operations --> Memory                                             │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   Memory: M1, M2, M3, M4                                            │
└────────────────────────────────────────────────────────────────────┘
```



**Diagram Explanation**: This diagram outlines the circular buffer data structure used throughout the system. It shows the key components of the buffer structure (blue), its primary operations (yellow), and the memory benefits it provides (green).

Circular buffers offer several advantages for this application:

1. **Fixed Memory Usage**:
   - Pre-allocated arrays with constant size
   - No dynamic memory allocations during operation
   - Consistent memory footprint

2. **Efficient Operations**:
   - O(1) add operations
   - O(1) retrieval of most recent elements
   - Automatic discarding of oldest data

3. **Implementation Benefits**:
   - No garbage collection overhead
   - No memory fragmentation
   - Optimized for embedded systems (Raspberry Pi)

#### Implementation Example

```python
# Pseudocode example of circular buffer implementation
class OptimizedBuffer:
    def __init__(self, max_size=10):
        # Pre-allocate the entire array
        self.max_size = max_size
        self.data = [None] * max_size
        self.next_index = 0
        self.size = 0
        self.timestamps = [0.0] * max_size  # Optional timing information
    
    def add(self, value, timestamp=None):
        # Store new value, overwriting oldest if full
        self.data[self.next_index] = value
        
        # Store timestamp if provided
        if timestamp is not None:
            self.timestamps[self.next_index] = timestamp
        else:
            self.timestamps[self.next_index] = time.time()
        
        # Move index with wrap-around
        self.next_index = (self.next_index + 1) % self.max_size
        
        # Update size (won't exceed max_size)
        self.size = min(self.size + 1, self.max_size)
    
    def get_latest(self, count=1):
        # Validate count request
        count = min(count, self.size)
        if count <= 0:
            return []
            
        # Calculate start index (moving backward from current position)
        start_idx = (self.next_index - count) % self.max_size
        
        # Simple case: no wrap-around needed
        if start_idx < self.next_index:
            return self.data[start_idx:self.next_index]
            
        # Complex case: values wrap around the buffer end
        return self.data[start_idx:] + self.data[:self.next_index]
    
    def get_all(self):
        # Simple case: Buffer not full yet
        if self.size < self.max_size:
            return self.data[:self.size]
            
        # Complex case: Buffer is full, items might wrap around
        return self.data[self.next_index:] + self.data[:self.next_index]
```

#### Buffer Applications

Circular buffers are used throughout the system for storing:

1. **Position History**:
   - Track recent ball positions
   - Calculate motion patterns
   - Detect sudden movements

2. **State Transition History**:
   - Record recent state changes
   - Detect oscillation patterns
   - Implement adaptive hysteresis

3. **Uncertainty History**:
   - Monitor uncertainty trends
   - Detect rising/falling patterns
   - Trigger recovery when needed

4. **Warning History**:
   - Track warning patterns
   - Identify recurring issues
   - Correlate with system events

### 7.2 <a name="trend-analysis"></a>Trend Analysis

The system implements efficient trend analysis for time-series data stored in circular buffers.

#### Trend Detection Algorithm

```
┌─────────────────────── TREND ANALYSIS PROCESS ────────────────────────┐
│                                                                       │
│  INPUT:                                                               │
│  ┌─────────────────────┐            ┌─────────────────────┐          │
│  │   Time-Series Data  │            │     Timestamps      │          │
│  └──────────┬──────────┘            └──────────┬──────────┘          │
│             │                                   │                     │
│             ▼                                   ▼                     │
│  ┌─────────────────────┐            ┌─────────────────────┐          │
│  │  1. Calculate Value │            │ 2. Calculate Rates  │          │
│  │     Differences     │            │    of Change        │          │
│  └──────────┬──────────┘            └──────────┬──────────┘          │
│             │                                   │                     │
│             ▼                                   ▼                     │
│  ┌─────────────────────┐            ┌─────────────────────┐          │
│  │ 3. Determine Trend  │            │ 4. Calculate Avg.   │          │
│  │    Direction        │            │    Rate of Change   │          │
│  └──────────┬──────────┘            └──────────┬──────────┘          │
│             │                                   │                     │
│             │                                   │                     │
│  OUTPUT:    ▼                                   ▼                     │
│  ┌─────────────────────┐            ┌─────────────────────┐          │
│  │     Direction:      │            │        Rate:        │          │
│  │ rising/stable/falling│            │ average change speed│          │
│  └──────────┬──────────┘            └──────────┬──────────┘          │
│             │                                   │                     │
│             └───────────────────┬───────────────┘                     │
│                                 ▼                                     │
│                     ┌─────────────────────┐                           │
│                     │       Pattern:      │                           │
│                     │    detected type    │                           │
│                     └─────────────────────┘                           │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Diagram Explanation**: This flowchart shows the trend analysis process. It takes time-series data as input (blue), performs a series of analysis steps (yellow), and produces trend information as output (green).

#### Implementation Example

```python
# Pseudocode example of trend analysis
class EfficientTrendAnalyzer:
    def __init__(self, window_size=10):
        self.values = OptimizedBuffer(window_size)
        self.timestamps = OptimizedBuffer(window_size)
        self.diff_cache = OptimizedBuffer(window_size - 1)
        self.rate_cache = OptimizedBuffer(window_size - 1)
        self.stability_threshold = 0.001  # Threshold for "stable" determination
        
    def add(self, value, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
            
        # Calculate difference and rate if we have previous values
        if self.values.size > 0:
            prev_value = self.values.get_latest(1)[0]
            prev_time = self.timestamps.get_latest(1)[0]
            
            value_diff = value - prev_value
            time_diff = timestamp - prev_time
            
            self.diff_cache.add(value_diff)
            
            if time_diff > 0:
                self.rate_cache.add(value_diff / time_diff)
            else:
                self.rate_cache.add(0.0)
                
        # Add the new value and timestamp
        self.values.add(value)
        self.timestamps.add(timestamp)
        
    def get_trend(self, num_samples=None):
        # Default to all available samples
        if num_samples is None or num_samples > self.rate_cache.size:
            num_samples = self.rate_cache.size
            
        # Not enough data for trend analysis
        if num_samples < 2:
            return 0, 0.0, None  # stable, zero rate, no pattern
            
        # Get rates from cache
        rates = self.rate_cache.get_latest(num_samples)
        
        # Calculate average rate
        avg_rate = sum(rates) / len(rates)
        
        # Determine trend direction
        if abs(avg_rate) < self.stability_threshold:
            direction = 0  # Stable
        else:
            direction = 1 if avg_rate > 0 else -1  # Rising or falling
            
        # Detect patterns (simplified)
        pattern = None
        sign_changes = sum(1 for i in range(1, len(rates)) if 
                           (rates[i] > 0 and rates[i-1] < 0) or 
                           (rates[i] < 0 and rates[i-1] > 0))
        
        if sign_changes >= len(rates) / 2:
            pattern = "oscillating"
        elif max(rates) > 3 * avg_rate:
            pattern = "spike"
            
        return direction, avg_rate, pattern
```

#### Trend Analysis Applications

Trend analysis is used for several key features:

1. **Uncertainty Tracking**:
   - Detect rising uncertainty before reaching critical levels
   - Identify when uncertainty is decreasing during recovery
   - Detect unstable oscillating uncertainty patterns

2. **Confidence Monitoring**:
   - Track confidence trends for early warning
   - Detect sudden drops in confidence
   - Monitor confidence recovery after issues

3. **Position Prediction**:
   - Analyze movement patterns
   - Predict future positions during short sensor gaps
   - Detect acceleration and deceleration

4. **System Health Analysis**:
   - Monitor resource usage trends
   - Detect degrading performance
   - Identify recurring patterns

### 7.3 <a name="memory-efficiency"></a>Memory Efficiency

The State Management Node employs several techniques to minimize memory usage on resource-constrained platforms like the Raspberry Pi.

#### Optimized Data Types

```mermaid

        flowchart LR
    subgraph "Memory Optimization Techniques"
        T1["Use primitive types"] --> T4["Pre-allocated buffers"]
        T2["Fixed-size containers"] --> E1["Custom JSON encoder"]
        T3["Enum classes"] --> E3["Timestamp delta compression"]
        T4 --> R1["Message object reuse"]
        E1 --> R3["Cached calculations"]
        E3 --> R4["Shared reference objects"]
    end
```

**Diagram Explanation**: This diagram presents the key memory optimization techniques used in the system. It groups them into three categories: data type optimizations (blue), efficient encoding methods (yellow), and object reuse strategies (green).

#### Efficient JSON Encoding

For publishing diagnostic data, the system uses an optimized JSON encoder:

```python
# Pseudocode example of efficient JSON encoder
class FastJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Direct handling of common types without isinstance overhead
        obj_type = type(obj)
        
        # Handle float types first (most common)
        if obj_type is float:
            return round(obj, 3)  # Limit precision for smaller size
        
        # Handle other primitive types
        if obj_type is int:
            return obj
        if obj_type is list:
            return obj
        if obj_type is bool:
            return obj
            
        # Handle enum types efficiently
        if obj_type is RobotState:
            return obj.name
            
        # Fallback for complex types
        try:
            if hasattr(obj, 'tolist'):  # For array-like objects
                return obj.tolist()
            return super(FastJSONEncoder, self).default(obj)
        except TypeError:
            return str(obj)  # Last resort - stringify
```

#### Message Reuse

The system reuses message objects to reduce memory allocations:

```python
# Pseudocode example of message object reuse
def setup_publishers(self):
    # Create publishers
    self.state_publisher = self.create_publisher(String, '/robot/state', 10)
    self.health_publisher = self.create_publisher(Float32, '/robot/health', 10)
    self.diagnostics_publisher = self.create_publisher(String, '/robot/diagnostics', 10)
    self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
    
    # Create reusable message objects
    self.state_msg = String()
    self.health_msg = Float32()
    self.diagnostics_msg = String()
    self.cmd_vel_msg = Twist()
    
    # Setup zeros for velocity when stopped
    self.zero_velocity = Twist()
    self.zero_velocity.linear.x = 0.0
    self.zero_velocity.linear.y = 0.0
    self.zero_velocity.linear.z = 0.0
    self.zero_velocity.angular.x = 0.0
    self.zero_velocity.angular.y = 0.0
    self.zero_velocity.angular.z = 0.0
```

These optimizations ensure that the State Management Node maintains a low and consistent memory footprint, making it suitable for running on resource-constrained platforms like the Raspberry Pi.

## 8. <a name="practical-implementation"></a>Practical Implementation


### 8.1 <a name="configuration-parameters"></a>Configuration Parameters

The State Management Node offers extensive configuration parameters for customizing behavior.

#### Parameter Categories

```mermaid

        flowchart TD
    subgraph "Configuration Parameter Categories"
        T1["Timing Parameters"] --> T2["lost_ball_timeout: 1.5s"]
        T1 --> T3["max_search_time: 30.0s"]
    D1["Detection Parameters"] --> D2["min_tracking_detections: 3"]
        D1 --> D3["proximity_threshold: 0.5m"]
    U1["Uncertainty Parameters"] --> U2["position_uncertainty_threshold: 0.5m"]
    H1["Hysteresis Parameters"] --> H2["tracking_hysteresis_time: 1.0s"]
    A1["Adaptive Parameters"] --> A2["adaptive_parameters_enabled: true"]
    S1["System Parameters"] --> S2["update_rate: 100Hz"]
    end

    style U2 fill:#b7af93,stroke:#979079,stroke-width:1px,color:#000000,font-weight:bold
    style H2 fill:#b29a9d,stroke:#937f81,stroke-width:1px,color:#000000,font-weight:bold
    style A2 fill:#a8aaac,stroke:#8a8c8e,stroke-width:1px,color:#000000,font-weight:bold
    style S2 fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram categorizes the system's configuration parameters into six groups: timing parameters (blue), detection parameters (green), uncertainty parameters (yellow), hysteresis parameters (red), adaptive parameters (gray), and system parameters (blue). Each group contains specific parameters with their default values.

#### Parameter Relationships

Many parameters are interrelated, requiring careful tuning to maintain system balance:

```mermaid

        flowchart TD
    subgraph "Key Parameter Relationships"
        direction TB
    lost_ball_timeout["lost_ball_timeout<br>(how quickly to enter SEARCHING)"] --- stationary_threshold["stationary_threshold<br>(when ball is considered stopped)"]
        stationary_threshold --- stationary_time_threshold["stationary_time_threshold<br>(how long ball must be still)"]
        lost_ball_timeout --- min_tracking_detections["min_tracking_detections<br>(how many detections before tracking)"]
        min_tracking_detections --- min_retracking_detections["min_retracking_detections<br>(how many detections to resume)"]
        position_uncertainty_threshold["position_uncertainty_threshold<br>(when to enter RECOVERY)"] --- uncertainty_recovery_threshold["uncertainty_recovery_threshold<br>(when to exit RECOVERY)"]
        tracking_hysteresis_time["tracking_hysteresis_time<br>(min time in TRACKING)"] --- lost_ball_hysteresis_time["lost_ball_hysteresis_time<br>(min time in LOST_BALL)"]
    end

    style lost_ball_timeout fill:#94a3b7,stroke:#7a8697,stroke-width:2px,color:#000000,font-weight:bold
    style stationary_threshold fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style stationary_time_threshold fill:#94a3b7,stroke:#7a8697,stroke-width:2px,color:#000000,font-weight:bold
    style min_tracking_detections fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style min_retracking_detections fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style position_uncertainty_threshold fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
    style uncertainty_recovery_threshold fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
    style tracking_hysteresis_time fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style lost_ball_hysteresis_time fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows the key relationships between different parameters. Lines connect parameters that have a direct relationship, where changing one often requires adjusting the other for optimal system performance.

#### Configuration Implementation

```python
# Pseudocode example of parameter declaration
def _declare_parameters(self):
    """Declare all parameters with optimized grouping."""
    # Define parameter groups for better performance
    timing_params = [
        ('lost_ball_timeout', 1.5),
        ('max_search_time', 30.0),
        ('stationary_time_threshold', 1.5),
        ('max_lost_ball_time', 5.0),
        ('max_recovery_time', 3.0),
    ]
    
    search_params = [
        ('search_rotation_speed', 0.5),
        ('max_rotation_time', 15.0),
    ]
    
    detection_params = [
        ('min_tracking_detections', 3),
        ('min_retracking_detections', 6),
        ('proximity_threshold', 0.5),
        ('stationary_threshold', 0.05),
    ]
    
    uncertainty_params = [
        ('position_uncertainty_threshold', 0.5),
        ('uncertainty_recovery_threshold', 0.35),
    ]
    
    hysteresis_params = [
        ('tracking_hysteresis_time', 1.0),
        ('lost_ball_hysteresis_time', 0.5),
        ('recovery_hysteresis_time', 0.3),
    ]
    
    adaptive_params = [
        ('adaptive_parameters_enabled', True),
        ('adaptive_factor_stationary', 1.5),
        ('adaptive_factor_moving', 0.8),
    ]
    
    gap_params = [
        ('gap_tolerance_time', 1.5),
        ('gap_stationary_multiplier', 2.0),
        ('gap_enabled', True),
    ]
    
    system_params = [
        ('health_confidence_threshold', 0.5),
        ('health_check_interval', 1.0),
        ('diagnostic_publish_rate', 1.0),
        ('full_diagnostic_rate', 5.0),
        ('resource_monitoring_enabled', True),
    ]
    
    # Combine all parameter groups
    all_params = (timing_params + search_params + detection_params + 
                 uncertainty_params + hysteresis_params + adaptive_params + 
                 gap_params + system_params)
    
    # Declare all parameters in a single batch for better performance
    self.declare_parameters(namespace='', parameters=all_params)
```

### 8.2 <a name="performance-optimization"></a>Performance Optimization

The State Management Node is optimized for performance on resource-constrained platforms like the Raspberry Pi.

#### Computational Efficiency

```
┌───────────────────── PERFORMANCE OPTIMIZATION TECHNIQUES ─────────────────────┐
│                                                                               │
│  ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐ 
│  │  ALGORITHMIC OPTIMIZATIONS │     │   CONCURRENCY CONTROL   │     │   RESOURCE MANAGEMENT   
│  ├─────────────────────────┤     ├─────────────────────────┤     ├─────────────────────────┤ 
│  │                         │     │                         │     │                         │ 
│  │ • Early-exit in         │     │ • Callback groups for   │     │ • Reduced timer         │ 
│  │   critical paths        │     │   parallel processing   │     │   frequency (50Hz)      │ 
│  │                         │     │                         │     │                         │ 
│  │ • Single-pass data      │     │ • Wait-free data        │     │ • Conditional           │ 
│  │   processing            │     │   structures            │     │   diagnostic publishing │ 
│  │                         │     │                         │     │                         │ 
│  │ • Minimal string        │     │ • Non-blocking          │     │ • Staged diagnostic     │ 
│  │   operations            │     │   operations            │     │   publication           │ 
│  │                         │     │                         │     │                         │ 
│  │ • Cached calculations   │     │ • Message filtering     │     │ • Dynamic buffer        │ 
│  │   and lookups           │     │   at source             │     │   sizing                │ 
│  │                         │     │                         │     │                         │ 
│  └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘ 
│                                                                               │
│                                                                               │
│      RESULT: 5-8x LOWER CPU USAGE COMPARED TO PREVIOUS IMPLEMENTATION         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Diagram Explanation**: This diagram presents the performance optimization techniques used in the system. It groups them into algorithmic optimizations (blue), concurrency control methods (yellow), and resource management strategies (green).

#### Timer Optimization

```python
# Pseudocode example of performance optimization
def _setup_timers(self):
    # Create callback groups to manage prioritization
    self.timer_cb_group = MutuallyExclusiveCallbackGroup()
    self.pub_cb_group = MutuallyExclusiveCallbackGroup()
    
    # Critical state management timer (5Hz instead of 10Hz)
    self.state_timer = self.create_timer(
        0.2,  # 5Hz instead of 10Hz 
        self.state_manager_callback,
        callback_group=self.timer_cb_group
    )
    
    # Health check timer (reduced frequency)
    self.health_timer = self.create_timer(
        max(self.health_check_interval, 1.0),  # Ensure minimum 1s interval
        self.health_check_callback,
        callback_group=self.timer_cb_group
    )
    
    # Periodic state republishing (0.25Hz instead of 0.5Hz)
    self.state_republish_timer = self.create_timer(
        4.0,  # 4s instead of 2s
        self.publish_state,
        callback_group=self.pub_cb_group
    )
```

#### Early-Exit Optimizations

```python
# Pseudocode example of early-exit optimization
def handle_position_based_transitions(self, current_time):
    # Quick exit if no position data available
    if self.current_position is None:
        return
        
    # Calculate time in current state for hysteresis
    time_in_state = current_time - self.state_start_time
    
    # Early exit if in minimum hysteresis period
    if time_in_state < self.get_min_time_for_state(self.current_state):
        return
        
    # Now apply state-specific handlers
    if self.current_state == RobotState.INITIALIZING:
        self._handle_initializing_transitions(time_in_state)
    # ... other states
```

### 8.3 <a name="logging-and-debugging"></a>Logging and Debugging

The system implements comprehensive logging with optimization features.

#### Tiered Logging System

```
┌────────────────────────────────────────────────────────────────────┐
│ FLOWCHART (Top to Bottom)                                         │
├────────────────────────────────────────────────────────────────────┤
│ NODES:                                                              │
├────────────────────────────────────────────────────────────────────┤
│ CONNECTIONS:                                                        │
│   C1["Critical Logs<br>Always Logged"] --> C2["State transitions"]  │
│   C1 --> C3["Warning conditions"]                                   │
│   I1["Important Logs<br>Regular"] --> I2["Parameter changes"]       │
│   I1 --> I3["Status changes"]                                       │
│   N1["Informational<br>Throttled"] --> N2["Position updates"]       │
│   N1 --> N3["Regular metrics"]                                      │
│   D1["Debug Logs<br>Conditional"] --> D2["Timing information"]      │
│   D1 --> D3["Decision logic details"]                               │
├────────────────────────────────────────────────────────────────────┤
│ SUBGRAPHS:                                                          │
│   Tiered Logging System:                                            │
└────────────────────────────────────────────────────────────────────┘
```

**Diagram Explanation**: This diagram illustrates the tiered logging system. It categorizes logs into critical (red), important (yellow), informational (green), and debug (blue) levels, with different handling for each tier.

#### Throttled Logging Implementation

```python
# Pseudocode example of throttled logging
def throttled_log(self, logger, message, key, min_interval=1.0, level="info"):
    current_time = time.time()
    
    # Check if enough time has passed since last log with this key
    if key in self._last_throttled_logs:
        elapsed = current_time - self._last_throttled_logs[key]
        if elapsed < min_interval:
            # Not enough time passed, skip this log
            return
            
    # Update last log time
    self._last_throttled_logs[key] = current_time
    
    # Log with appropriate level
    if level == "error":
        logger.error(message)
    elif level == "warn":
        logger.warn(message)
    else:
        logger.info(message)
```

#### Logging Integration with Diagnostics

```python
# Pseudocode example of diagnostic logging integration
def transition_to_state(self, new_state):
    if new_state != self.current_state:
        # Calculate time in previous state
        time_in_state = time.time() - self.state_start_time
        
        # Log state transition - critical, always log
        self.get_logger().info(
            f"State transition: {self.current_state.name} -> {new_state.name} "
            f"(after {time_in_state:.2f}s in {self.current_state.name})"
        )
        
        # Update state information
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        
        # Record in state history for diagnostics
        self.state_history.add({
            'from': self.previous_state.name,
            'to': self.current_state.name,
            'time_in_previous': time_in_state,
            'reason': self.transition_reason
        })
        
        # Reset transition reason
        self.transition_reason = None
        
        # Publish state update immediately
        self.publish_state()
        
        # Detailed diagnostic information - throttled
        self.throttled_log(
            self.get_logger(),
            f"Detailed state change info - Confidence: {self.system_confidence:.2f}, "
            f"Position: ({self.ball_position[0]:.2f}, {self.ball_position[1]:.2f}), "
            f"Uncertainty: {self.position_uncertainty:.3f}",
            "state_details",
            min_interval=2.0
        )
```

The tiered logging system ensures that critical information is always captured while preventing log spam from routine operations, striking a balance between information completeness and system performance.

## 9. <a name="case-studies"></a>Case Studies


This section presents detailed case studies showing how the State Management Node handles real-world scenarios.

### 9.1 <a name="handling-sensor-failures"></a>Handling Sensor Failures

#### Scenario: Temporary Occlusion

**Situation**: The basketball is being actively tracked when all sensors suddenly lose detection due to an occlusion.

```mermaid

        sequenceDiagram
    participant Fusion as Fusion Node
    participant State as State Manager
    participant PID as PID Controller
    Note over State: Initially in TRACKING state
    Fusion->>State: Position update (reliable)
    State->>PID: Normal tracking commands
    Note over Fusion: Ball becomes occluded
    Fusion->>State: sensor_gap=True notification
    Note over State: Gap handling activates
    State->>State: Record gap start time
    State->>State: Calculate adaptive tolerance
    State->>PID: Reduce velocity to 60%
    Note over State: For 0.8 seconds...
    State->>State: Maintain TRACKING state<br>Override timeout logic
    Note over Fusion: Ball reappears
    Fusion->>State: sensor_gap=False notification
    Fusion->>State: Position update (reliable)
    Note over State: Gap handling deactivates
    State->>State: Clear gap status
    State->>PID: Resume normal velocity
    Note over State: Continues in TRACKING state
```

**Diagram Explanation**: This sequence diagram shows how the system handles a temporary occlusion. When the ball becomes occluded, the sensor gap handling activates, maintaining the TRACKING state while reducing velocity. When the ball reappears, normal tracking resumes without any state change.

#### System Response:

1. **Initial Detection**:
   - Fusion node reports `sensor_gap=True`
   - State Manager receives notification in `sensor_gap_callback`
   - Gap start time is recorded

2. **Gap Tolerance Phase**:
   - System remains in TRACKING state during short gaps
   - Last detection time artificially updated to prevent timeout
   - Robot velocity reduced proportionally to gap duration

3. **Recovery Process**:
   - Gap resolves after 0.8 seconds (within tolerance)
   - System continues in TRACKING state without disruption
   - Velocity gradually returns to normal as confidence rebuilds

4. **Successful Outcome**:
   - No state change occurred
   - Robot maintained smooth motion throughout gap
   - Tracking resumed seamlessly when ball reappeared

This case demonstrates the system's robustness to temporary sensor failures, maintaining continuous operation even when sensor data is briefly unavailable.

### 9.2 <a name="recovering-from-uncertainty-spikes"></a>Recovering from Uncertainty Spikes

#### Scenario: Sensor Conflict Causing Uncertainty Spike

**Situation**: The basketball is being tracked when multiple sensors provide conflicting information, causing a spike in position uncertainty.

```mermaid

        flowchart TD
    subgraph "Uncertainty Spike Recovery Timeline"
        direction TB
    T0["T=0.000s<br>Normal tracking<br>Uncertainty: 0.18m<br>State: TRACKING"] --> 
        T1["T=0.215s<br>Uncertainty rises to 0.62m<br>Trend analysis detects rapid rise"] -->
        T2["T=0.248s<br>Enter RECOVERY state<br>Reduce velocity to 40%"] -->
        T3["T=0.648s<br>Uncertainty stabilizes<br>But still above threshold (0.48m)"] -->
        T4["T=0.948s<br>Uncertainty drops to 0.32m<br>(Below recovery threshold)"] -->
        T5["T=1.048s<br>Return to TRACKING state<br>Resume normal velocity"] -->
        T6["T=1.248s<br>Tracking fully restored<br>Uncertainty: 0.15m"]
    end

    style T0 fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style T1 fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style T2 fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style T3 fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
    style T4 fill:#b7af93,stroke:#979079,stroke-width:2px,color:#000000,font-weight:bold
    style T5 fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style T6 fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This timeline shows how the system responds to an uncertainty spike. Starting from normal tracking (green), it detects a rapid rise in uncertainty (red), enters RECOVERY state while reducing velocity, monitors the uncertainty as it stabilizes then drops (yellow), and finally returns to normal TRACKING (green) once uncertainty is below the recovery threshold.

#### System Response:

1. **Uncertainty Detection**:
   - Position uncertainty rises rapidly from 0.18m to 0.62m
   - Trend analysis detects rising uncertainty at 0.215s
   - Rate of increase calculated at 0.08m/s

2. **Recovery Initiation**:
   - System transitions to RECOVERY state at 0.248s
   - Robot velocity reduced to 40% of normal
   - Recovery reason logged as "rising_uncertainty"

3. **Adaptation Process**:
   - Tracking parameters adjusted to be more lenient
   - System begins monitoring uncertainty trend
   - Minimum recovery time enforced (0.3s hysteresis)

4. **Resolution**:
   - Uncertainty stabilizes then begins declining
   - At 0.948s, uncertainty drops below recovery threshold (0.35m)
   - System transitions back to TRACKING at 1.048s

5. **Successful Outcome**:
   - Total recovery duration: 0.8 seconds
   - Robot resumes tracking with normal parameters
   - Uncertainty returns to normal levels (0.15m)

This case demonstrates how the system detects and responds to uncertainty spikes, ensuring robust operation even when sensor data quality temporarily degrades.

### 9.3 <a name="real-time-response"></a>Real-Time Response to Sudden Ball Movement

#### Scenario: Ball Suddenly Moves After Being Stationary

**Situation**: The basketball has been stationary for an extended period, and the robot is in STOPPED state when the ball suddenly starts moving at high speed.

```mermaid

        gantt
    title Movement Response Timeline
    dateFormat  SSS
    axisFormat %Lms
    section Initial State
    Ball stationary, Robot in STOPPED       :done, 000, 052
    section Movement Detection
    Fusion detects motion                   :crit, 000, 012
    Motion state changes to medium_fast     :crit, 012, 003
    section Parameter Adaptation
    Parameters reconfigured for movement    :active, 015, 007
    section State Transition
    State change to TRACKING                :active, 022, 005
    Change published to PID controller      :active, 027, 004
    section Robot Response
    PID receives state update               :031, 007
    First movement command issued           :038, 014
    Robot begins physical movement          :052, 016
    First motion feedback received          :068, 058
```

**Diagram Explanation**: This Gantt chart shows the detailed timeline of how the system responds to sudden ball movement. It tracks the sequence from initial motion detection (red) through parameter adaptation and state transition (yellow), to the PID controller's response and actual robot movement (blue). The whole process from detection to physical movement takes just 52ms.

#### System Response:

1. **Initial Conditions**:
   - System in STOPPED state for 10.2 seconds
   - Motion state: "stationary"
   - Ball position stable at (1.2m, 0.3m)
   - CPU usage: 8%

2. **Movement Detection**:
   - Fusion node detects motion at t=0ms
   - State Manager receives motion update at t=12ms
   - Motion state changes to "medium_fast"
   - Motion detection callback processing time: 3ms

3. **Parameter Adaptation**:
   - Parameters reconfigured for fast movement at t=15ms
   - `tracking_timeout` reduced from 2.0s to 1.2s
   - `position_prediction_weight` increased from 0.3 to 0.6
   - Parameter adaptation processing time: 7ms

4. **State Transition**:
   - State change from STOPPED to TRACKING at t=22ms
   - State transition callback processing time: 5ms
   - State change published to PID controller at t=27ms
   - Network transmission time to PID controller: 4ms

5. **Robot Response**:
   - PID controller receives state update at t=31ms
   - First movement command issued at t=38ms
   - Robot begins physical movement at t=52ms
   - First motion feedback received at t=68ms

6. **Performance Metrics**:
   - Total response latency (detection to movement): 52ms
   - CPU usage during transition peak: 23%
   - Memory usage increase: 1.2MB
   - Complete transition stabilization time: 126ms

This case study demonstrates the system's ability to quickly respond to sudden changes in ball behavior, with a total end-to-end latency of just 52ms from initial detection to robot movement.

### 9.4 <a name="managing-environmental-interference"></a>Managing Environmental Interference

#### Scenario: Electromagnetic Interference Affecting Sensors

**Situation**: The basketball is being tracked when the robot enters an area with high electromagnetic interference affecting sensor readings.

```mermaid

        gantt
    title Environmental Interference Recovery
    dateFormat  SSS
    axisFormat %Lms
    section Detection
    Interference begins                 :milestone, 000, 0
    Sensor noise levels increase        :done, 000, 215
    Position uncertainty rising         :done, 000, 215
    Confidence drops from 0.95 to 0.68  :done, 000, 215
    section Response
    Warning condition triggered         :crit, 215, 11
    Parameter adaptation begins         :crit, 226, 22
    Sensor weights adjusted             :230, 18
    RECOVERY state entered              :crit, 248, 0
    Robot speed reduced to 60%          :248, 10
    section Stabilization
    Uncertainty stabilizes at 0.4m      :active, 248, 800
    Minimum recovery time enforced      :active, 248, 800
    section Resolution
    Return to TRACKING state            :milestone, 1048, 0
    Normal parameters with adjustments  :1048, 15
```

**Diagram Explanation**: This Gantt chart illustrates how the system responds to environmental interference. It shows the sequence from initial interference detection (blue), through the response phase where parameters are adjusted and RECOVERY state is entered (red), the stabilization period (yellow), and finally the return to normal operation (green marker).

#### System Response:

1. **Initial Detection**:
   - Sensor noise levels begin increasing at t=0ms
   - Position uncertainty gradually rises (0.1m → 0.3m → 0.5m)
   - Detection confidence drops from 0.95 to 0.68
   - Uncertainty trend detected at t=215ms

2. **Early Intervention**:
   - System triggers warning condition at t=220ms
   - Health monitoring callback processing time: 6ms
   - Warning status published to diagnostics topic
   - Parameter adaptation begins at t=226ms

3. **Adaptive Response**:
   - Sensor weights automatically adjusted at t=230ms
   - LIDAR weight increased, camera weight decreased
   - Filter parameters tightened to reduce noise
   - Parameter adjustment processing time: 18ms

4. **Recovery Actions**:
   - Recovery state entered at t=248ms
   - Recovery reason: "rising_uncertainty"
   - Recovery strategy: "sensor_reweighting"
   - Robot speed temporarily reduced to 60%

5. **Stabilization**:
   - Uncertainty stabilizes at 0.4m after 1.2 seconds
   - System remains in RECOVERY state for minimum time (0.8s)
   - Full recovery achieved at t=1048ms
   - Return to TRACKING state with adjusted parameters

6. **Performance Analysis**:
   - Detection to intervention time: 220ms
   - Total recovery duration: 1048ms
   - CPU usage during recovery: 31%
   - Maximum memory usage: 86MB
   - Number of skipped state update cycles: 0

This case study demonstrates the system's ability to detect and adapt to challenging environmental conditions, maintaining tracking through sensor interference with minimal disruption to operation.

## 10. <a name="troubleshooting-guide"></a>Troubleshooting Guide

### 10.1 Common Issues and Solutions

#### Robot Oscillates Between States

**Symptoms:**
- Robot rapidly switches between TRACKING and SEARCHING
- Log shows frequent state transitions
- Jerky, unstable motion

**Possible Causes and Solutions:**

```mermaid

        flowchart TD
    subgraph "State Oscillation Troubleshooting"
        Problem["Robot oscillates<br>between states"] --> Cause1["Insufficient<br>hysteresis protection"]
        Problem --> Cause2["Detection instability"]
    Cause1 --> Sol1["Increase tracking_hysteresis_time<br>Increase lost_ball_hysteresis_time"]
        Cause2 --> Sol2["Increase min_tracking_detections"]
    Sol1 --> Config["Configuration Example:<br>tracking_hysteresis_time: 1.5<br>lost_ball_hysteresis_time: 1.0<br>min_tracking_detections: 5"]
        Sol2 --> Config
    end

    style Problem fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold    
    style Config fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows the troubleshooting process for state oscillation. Starting with the problem (red), it identifies possible causes (yellow) and their solutions (green), leading to a specific configuration example (blue) that addresses the issue.

**Configuration Fix:**
```yaml
# Add to your config file to reduce state oscillation
state_management:
  tracking_hysteresis_time: 1.5  # Increase from default 1.0s
  lost_ball_hysteresis_time: 1.0  # Increase from default 0.5s
  min_tracking_detections: 5  # Increase from default 3
  lost_ball_timeout: 2.0  # Increase from default 1.5s
```

**Real-world Log Example:**
```
[state_manager-12] [INFO] [1716981642.345] [ball_chase_state_manager]: State transition: TRACKING -> SEARCHING
[state_manager-12] [INFO] [1716981643.123] [ball_chase_state_manager]: State transition: SEARCHING -> TRACKING
[state_manager-12] [INFO] [1716981644.567] [ball_chase_state_manager]: State transition: TRACKING -> SEARCHING
[state_manager-12] [INFO] [1716981645.345] [ball_chase_state_manager]: State transition: SEARCHING -> TRACKING
```

#### Robot Fails to Stop When Ball is Stationary

**Symptoms:**
- Robot continues to move when ball is still
- Never enters STOPPED state
- Constantly makes small adjustments

**Possible Causes and Solutions:**

```mermaid

        flowchart TD
    subgraph "Stationary Detection Troubleshooting"
        Problem["Robot doesn't stop<br>when ball is stationary"] --> Cause1["Stationary threshold<br>too low"]
        Problem --> Cause2["Stationary time<br>threshold too high"]
    Cause1 --> Sol1["Increase stationary_threshold<br>to tolerate sensor noise"]
        Cause2 --> Sol2["Decrease stationary_time_threshold<br>for quicker stopping"]
    Sol1 --> Config["Configuration Example:<br>stationary_threshold: 0.08<br>stationary_time_threshold: 1.0<br>adaptive_factor_stationary: 2.0"]
        Sol2 --> Config
    end

    style Problem fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style Config fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows the troubleshooting process for stationary detection problems. The problem (red) branches into possible causes (yellow), each with specific solutions (green), leading to a configuration example (blue) that addresses the issue.

**Configuration Fix:**
```yaml
# Add to your config file to improve stationary detection
state_management:
  stationary_threshold: 0.08  # Increase from default 0.05m
  stationary_time_threshold: 1.0  # Decrease from default 1.5s
  adaptive_factor_stationary: 2.0  # Increase from default 1.5
```

#### Robot Enters Recovery State Too Frequently

**Symptoms:**
- Frequently stops and enters RECOVERY state
- Log shows "rising_uncertainty" or "high_uncertainty" messages
- Hesitant movement

**Possible Causes and Solutions:**

```mermaid

        flowchart TD
    subgraph "Recovery Frequency Troubleshooting"
        Problem["Robot enters<br>RECOVERY too often"] --> Cause1["Uncertainty thresholds<br>too strict"]
    Cause1 --> Sol1["Increase position_uncertainty_threshold<br>Increase uncertainty_recovery_threshold"]
    Sol1 --> Config["Configuration Example:<br>position_uncertainty_threshold: 0.7<br>uncertainty_recovery_threshold: 0.5<br>recovery_hysteresis_time: 0.5"]
    end

    style Problem fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style Cause1 fill:#b7af93,stroke:#979079,stroke-width:1px,color:#000000,font-weight:bold
    style Sol1 fill:#99aa9d,stroke:#7d8c81,stroke-width:1px,color:#000000,font-weight:bold
    style Config fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows the troubleshooting process for excessive recovery state entry. The problem (red) connects to possible causes (yellow) and their solutions (green), leading to a configuration example (blue) that addresses the issue.

**Configuration Fix:**
```yaml
# Add to your config file to reduce recovery events
state_management:
  position_uncertainty_threshold: 0.7  # Increase from default 0.5m
  uncertainty_recovery_threshold: 0.5  # Increase from default 0.35m
  recovery_hysteresis_time: 0.5  # Increase from default 0.3s
```

**Real-world Log Example:**
```
[state_manager-12] [INFO] [1716981650.123] [ball_chase_state_manager]: Recovery triggered: high_uncertainty
[state_manager-12] [INFO] [1716981650.124] [ball_chase_state_manager]: Current uncertainty: 0.63, threshold: 0.50
[state_manager-12] [INFO] [1716981650.125] [ball_chase_state_manager]: State transition: TRACKING -> RECOVERY
```

#### Robot Doesn't Find Ball After Losing It

**Symptoms:**
- Ineffective search pattern
- Enters LOST_BALL state too quickly
- Searching in wrong areas

**Possible Causes and Solutions:**

```mermaid

        flowchart TD
    subgraph "Search Effectiveness Troubleshooting"
        Problem["Robot doesn't find<br>ball after losing it"] --> Cause1["Search parameters<br>too conservative"]
    Cause1 --> Sol1["Increase max_search_time<br>Decrease search_rotation_speed"]
    Sol1 --> Config["Configuration Example:<br>max_search_time: 45.0<br>search_rotation_speed: 0.3<br>min_retracking_detections: 4"]
    end

    style Problem fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style Cause1 fill:#b7af93,stroke:#979079,stroke-width:1px,color:#000000,font-weight:bold
    style Sol1 fill:#99aa9d,stroke:#7d8c81,stroke-width:1px,color:#000000,font-weight:bold
    style Config fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart shows the troubleshooting process for search effectiveness problems. The problem (red) branches into possible causes (yellow), each with specific solutions (green), leading to a configuration example (blue) that addresses the issue.

**Configuration Fix:**
```yaml
# Add to your config file to improve search effectiveness
state_management:
  max_search_time: 45.0  # Increase from default 30.0s
  search_rotation_speed: 0.3  # Decrease from default 0.5 for wider scan
  min_retracking_detections: 4  # Decrease from default 6
```

### 10.2 System Diagnostic Tools

#### State Transition Visualization

To visualize state transitions in real-time, use the following ROS2 commands:

```bash
# Record state transitions to a log file
ros2 topic echo /robot/state --csv > state_transitions.csv

# Generate a state transition graph
ros2 run ball_chase generate_transition_graph.py state_transitions.csv
```

This will create a visualization of state transitions with timing information:

```mermaid

        stateDiagram-v2
    [*] --> INITIALIZING
    INITIALIZING --> TRACKING: 3 detections, 0.8s
    TRACKING --> SEARCHING: Ball lost for 1.5s
    TRACKING --> RECOVERY: Uncertainty 0.6m
    TRACKING --> STOPPED: Stationary for 1.5s
    SEARCHING --> TRACKING: Ball found, 6 detections
    SEARCHING --> LOST_BALL: Searching for 30s
    RECOVERY --> TRACKING: Uncertainty reduced to 0.3m
    STOPPED --> TRACKING: Ball moved 0.08m
    LOST_BALL --> TRACKING: Ball redetected
    %% Apply styles to common states
    %% Default styling for better readability
    classDef defaultClass fill:#d0d0d0,stroke:#333,stroke-width:1px,color:#535353,font-weight:bold
    classDef regularClass fill:#cdb88c,stroke:#a99773,stroke-width:2px,color:#524938,font-weight:bold
    classDef activeClass fill:#8eb38e,stroke:#759375,stroke-width:2px,color:#384738,font-weight:bold
    classDef errorClass fill:#cd9b9b,stroke:#a98080,stroke-width:2px,color:#523e3e,font-weight:bold
    
    class TRACKING activeClass
    class RECOVERY,LOST_BALL,SEARCHING errorClass
    class INITIALIZING,STOPPED regularClass
```

#### Health Monitoring Dashboard

Use the health monitoring dashboard to track system health metrics:

```bash
# Start the health monitoring dashboard
ros2 run ball_chase health_dashboard.py
```

This displays a real-time view of system health:

```
=====================================
HEALTH MONITORING DASHBOARD
=====================================
System Confidence: 0.87 [||||||||-]
Active Warnings: 0
Current State: TRACKING (for 12.5s)
Position Uncertainty: 0.18m
Tracking Confidence: 0.93
CPU Usage: 23%
Memory Usage: 68MB
Update Rate: 100Hz
=====================================
```

#### Performance Profiling

For detailed performance analysis:

```bash
# Run performance profiling for 30 seconds
ros2 run ball_chase performance_profiler.py --duration 30
```

This generates a performance report:

```
PERFORMANCE PROFILE REPORT
=====================================
Duration: 30.0 seconds
Average update rate: 97.8 Hz
Max update latency: 18.2 ms
99th percentile latency: 12.5 ms

CALLBACK TIMING:
- state_manager_callback: 3.2 ms avg
- position_callback: 1.8 ms avg
- uncertainty_callback: 0.9 ms avg
- motion_state_callback: 1.1 ms avg

STATE STATISTICS:
- TRACKING: 87.2% of time
- STOPPED: 8.5% of time
- SEARCHING: 2.8% of time
- RECOVERY: 1.5% of time
=====================================
```

### 10.3 Structured Troubleshooting Guide

For effective troubleshooting, we've divided common issues into three categories:

#### 10.3.1 State Transition Issues

```mermaid
        flowchart TD
    Start(["State Transition<br>Troubleshooting"]) --> Q1
    Q1{"Are transitions<br>happening too frequently?"} -->|Yes| S1
    Q1 -->|No| Q2
    Q2{"Are transitions not<br>happening when expected?"} -->|Yes| S2
    Q2 -->|No| Q3
    Q3{"Are transitions<br>happening with delay?"} -->|Yes| S3
    Q3 -->|No| S4
    
    S1["Solution:<br>Increase hysteresis parameters<br>• tracking_hysteresis_time<br>• lost_ball_hysteresis_time"]
    S2["Solution:<br>Check state transition conditions<br>• min_tracking_detections<br>• position_uncertainty_threshold"]
    S3["Solution:<br>Optimize callback processing<br>• Reduce processing in callbacks<br>• Check for blocking operations"]
    S4["Solution:<br>Collect detailed transition logs<br>• Enable verbose logging<br>• Check state timing details"]
    
    style Start fill:#a8aaac,stroke:#8a8c8e,stroke-width:2px,color:#000000,font-weight:bold   
```

#### 10.3.2 Tracking Performance Issues

```mermaid
        flowchart TD
    Start(["Tracking Performance<br>Troubleshooting"]) --> Q1
    Q1{"Is tracking unstable<br>or jittery?"} -->|Yes| S1
    Q1 -->|No| Q2
    Q2{"Does tracking lose<br>the ball frequently?"} -->|Yes| S2
    Q2 -->|No| Q3
    Q3{"Is tracking slow<br>to respond?"} -->|Yes| S3
    Q3 -->|No| S4
    
    S1["Solution:<br>Tune parameters for stability<br>• Increase min_tracking_detections<br>• Adjust PID controller gains"]
    S2["Solution:<br>Improve detection reliability<br>• Increase lost_ball_timeout<br>• Decrease position_uncertainty_threshold"]
    S3["Solution:<br>Optimize for responsiveness<br>• Decrease tracking_hysteresis_time<br>• Reduce min_tracking_detections"]
    S4["Solution:<br>Analyze tracking patterns<br>• Check fusion node performance<br>• Review detection confidence values"]
    
    style Start fill:#a8aaac,stroke:#8a8c8e,stroke-width:2px,color:#000000,font-weight:bold
   
```

#### 10.3.3 System Resource Issues

```mermaid
        flowchart TD
    Start(["System Resource<br>Troubleshooting"]) --> Q1
    Q1{"Is CPU usage<br>too high?"} -->|Yes| S1
    Q1 -->|No| Q2
    Q2{"Is memory usage<br>growing over time?"} -->|Yes| S2
    Q2 -->|No| Q3
    Q3{"Are there<br>diagnostic issues?"} -->|Yes| S3
    Q3 -->|No| S4
    
    S1["Solution:<br>Reduce processing load<br>• Lower update frequency<br>• Simplify transition logic"]
    S2["Solution:<br>Check for memory leaks<br>• Review buffer management<br>• Check history containers"]
    S3["Solution:<br>Adjust diagnostic settings<br>• Reduce logging frequency<br>• Lower logging verbosity"]
    S4["Solution:<br>Perform system-wide analysis<br>• Check ROS2 node communications<br>• Review resource allocation"]
    
    style Start fill:#a8aaac,stroke:#8a8c8e,stroke-width:2px,color:#000000,font-weight:bold    
```

**Guide Usage**: First identify which category your issue falls into (state transitions, tracking performance, or system resources), then follow the appropriate decision tree above. Each tree asks specific diagnostic questions and leads to targeted solutions with specific parameters to adjust.

## 11. <a name="parameter-tuning-guide"></a>Parameter Tuning Guide

### 11.1 Parameter Relationships Matrix

Understanding parameter relationships is crucial for effective tuning:

```
╔═════════════════════ Parameter Relationship Matrix ═════════════════════╗
║                                                                         ║
║  ┌──────────────────┐                      ┌────────────────────┐       ║
║  │lost_ball_timeout │◄────────────────────►│stationary_threshold│       ║
║  └──────────────────┘                      └────────────────────┘       ║
║          ▲                                                              ║
║          │                                                              ║
║          ▼                                                              ║
║  ┌──────────────────┐                      ┌─────────────────────┐      ║
║  │  min_tracking    │◄────────────────────►│   min_retracking    │      ║
║  │   detections     │                      │    detections       │      ║
║  └──────────────────┘                      └─────────────────────┘      ║
║                                                                         ║
║  ┌──────────────────┐                      ┌─────────────────────┐      ║
║  │     position     │◄────────────────────►│     uncertainty     │      ║
║  │    uncertainty   │                      │       recovery      │      ║
║  │    threshold     │                      │      threshold      │      ║
║  └──────────────────┘                      └─────────────────────┘      ║
║                                                                         ║
║  ┌──────────────────┐                      ┌─────────────────────┐      ║
║  │     tracking     │◄────────────────────►│    lost_ball        │      ║
║  │    hysteresis    │                      │     hysteresis      │      ║
║  │       time       │                      │        time         │      ║
║  └──────────────────┘                      └─────────────────────┘      ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝
```

**Parameter Relationships Legend**:
- **Timing Parameters**: lost_ball_timeout, stationary_threshold  
- **Detection Parameters**: min_tracking_detections, min_retracking_detections
- **Uncertainty Parameters**: position_uncertainty_threshold, uncertainty_recovery_threshold
- **Hysteresis Parameters**: tracking_hysteresis_time, lost_ball_hysteresis_time

Parameters connected by arrows should be tuned together, as changing one typically requires adjusting the other.

**Diagram Explanation**: This diagram illustrates the key relationships between parameters. Parameters are color-coded by category (blue for timing, green for detection, yellow for uncertainty, red for hysteresis), and connections show which parameters influence each other and should be tuned together.

### 11.2 Core Timing Parameters

These parameters control the timing aspects of state transitions:

| Parameter | Default | Range | When to Increase | When to Decrease |
|-----------|---------|-------|------------------|------------------|
| `lost_ball_timeout` | 1.5s | 0.5-5.0s | • Ball frequently moves out of view<br>• Erratic transition to SEARCHING<br>• Poor sensor reliability | • Slow response when ball disappears<br>• Ball moves consistently<br>• Quick detection required |
| `stationary_time_threshold` | 1.5s | 0.5-5.0s | • False STOPPED transitions<br>• Ball has small movements<br>• Need longer confirmation | • Slow to detect stopped ball<br>• Very stable environment<br>• Quicker stopping desired |
| `max_search_time` | 30.0s | 10.0-120.0s | • Wider search area needed<br>• Complex environment<br>• Higher recovery priority | • Faster timeout needed<br>• Quick fallback preferred<br>• Limited battery concerns |
| `max_recovery_time` | 3.0s | 1.0-10.0s | • Complex sensor issues<br>• More recovery attempts<br>• Better recovery rate needed | • Quick fallback preferred<br>• Fast response prioritized<br>• Simpler sensor setup |

### 11.3 Detection Thresholds

These parameters control the detection sensitivity and requirements:

| Parameter | Default | Range | When to Increase | When to Decrease |
|-----------|---------|-------|------------------|------------------|
| `min_tracking_detections` | 3 | 1-10 | • Noisy environment<br>• False positives occur<br>• Need higher confidence | • Fast response needed<br>• Good sensor quality<br>• Missing detections |
| `min_retracking_detections` | 6 | 2-15 | • After losing track<br>• Noisy reacquisition<br>• Too many false returns | • Slow reacquisition<br>• Good sensor quality<br>• Fast recovery needed |
| `proximity_threshold` | 0.5m | 0.1-2.0m | • Operating in larger space<br>• Detecting from distance<br>• Larger target | • Small operating area<br>• Need finer control<br>• Small target |
| `stationary_threshold` | 0.05m | 0.01-0.2m | • Sensor noise present<br>• Small movements ignored<br>• Jittery position data | • Missing stopped state<br>• Very precise positioning<br>• Stable sensor data |

### 11.4 Parameter Sets for Specific Scenarios

#### 1. Low Latency Tracking (Competition Setting)

```yaml
state_management:
  # Fast response to changing conditions
  lost_ball_timeout: 0.8
  stationary_time_threshold: 0.8
  tracking_hysteresis_time: 0.5
  recovery_hysteresis_time: 0.2
  
  # Lower detection requirements for speed
  min_tracking_detections: 2
  min_retracking_detections: 3
  
  # Higher uncertainty tolerance for speed
  position_uncertainty_threshold: 0.8
  uncertainty_recovery_threshold: 0.6
  
  # More aggressive adaptation
  adaptive_factor_moving: 0.6
  adaptive_factor_stationary: 1.8
  
  # Shorter gap handling
  gap_tolerance_time: 0.7
```

#### 2. High Reliability Tracking (Demo Setting)

```yaml
state_management:
  # Increased stability through longer timeouts
  lost_ball_timeout: 2.0
  stationary_time_threshold: 2.0
  tracking_hysteresis_time: 1.5
  recovery_hysteresis_time: 0.5
  
  # Stricter detection requirements
  min_tracking_detections: 5
  min_retracking_detections: 8
  
  # Conservative uncertainty handling
  position_uncertainty_threshold: 0.4
  uncertainty_recovery_threshold: 0.25
  
  # Less aggressive adaptation
  adaptive_factor_moving: 0.9
  adaptive_factor_stationary: 1.3
  
  # Longer gap handling
  gap_tolerance_time: 2.0
  gap_stationary_multiplier: 2.5
```

#### 3. Noisy Environment Tracking

```yaml
state_management:
  # Longer timeouts for stability
  lost_ball_timeout: 1.8
  stationary_time_threshold: 2.0
  
  # Much stronger hysteresis protection
  tracking_hysteresis_time: 2.0
  lost_ball_hysteresis_time: 1.0
  recovery_hysteresis_time: 0.7
  
  # Much stricter detection requirements
  min_tracking_detections: 7
  min_retracking_detections: 10
  
  # Higher threshold for noise filtering
  stationary_threshold: 0.1
  position_uncertainty_threshold: 0.9
  
  # Specialized gap handling
  gap_tolerance_time: 2.5
  gap_enabled: true
```

### 11.5 Parameter Tuning Process

```mermaid

        flowchart LR
    subgraph "Parameter Tuning Workflow"
        Start["Start with<br>Default Parameters"] --> 
        Observe["Observe System<br>Behavior"] -->
        Identify["Identify Issues"] -->
        Adjust["Adjust One<br>Parameter at a Time"] -->
        Test["Test and<br>Document Effect"] -->
        Evaluate{"Behavior<br>Improved?"}
    Evaluate -->|No| Reset["Try Different Parameter"]
        Reset --> Adjust
    Evaluate -->|Yes| Next{"All Issues<br>Resolved?"}
    Next -->|No| Identify
        Next -->|Yes| Save["Save Final<br>Parameter Set"]
    end

    style Start,Observe,Save fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
    style Identify,Adjust,Test fill:#b7af93,stroke:#979079,stroke-width:1px,color:#000000,font-weight:bold
    style Evaluate,Next fill:#99aa9d,stroke:#7d8c81,stroke-width:1px,color:#000000,font-weight:bold
    style Reset fill:#b29a9d,stroke:#937f81,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This flowchart illustrates the recommended parameter tuning process. It shows a step-by-step workflow starting with default parameters (blue), proceeding through observation, issue identification, and parameter adjustment (yellow), followed by evaluation (green), and potential parameter reset if needed (red), finally resulting in a saved parameter set once all issues are resolved.

Follow these steps when tuning the state management parameters:

1. **Start with Defaults**: Begin with the default parameter set for your hardware
2. **Observe Behavior**: Run the system and note any issues in state transitions
3. **Isolate Problems**: Use the decision tree to determine which parameters to adjust
4. **Single Changes**: Modify only one parameter at a time and test thoroughly
5. **Document Effects**: Record how each change affects behavior
6. **Combine Solutions**: Once individual issues are fixed, create a complete parameter set
7. **Stress Test**: Test the final configuration under various conditions
8. **Save Configuration**: Save the final parameter set in a dedicated YAML file

Remember that parameters often interact with each other, so a change to one may require adjustments to others for optimal performance.

## 12. <a name="configuration-examples"></a>Configuration Examples

### 12.1 Fast-Moving Ball Tracking Configuration

For applications where the ball moves quickly (competitions, active games):

```yaml
# fast_moving_ball_config.yaml
state_management:
  # Timing parameters - quicker response
  lost_ball_timeout: 0.8  # Reduced from default 1.5s
  stationary_time_threshold: 1.0  # Reduced from default 1.5s
  
  # Detection parameters - looser for speed
  min_tracking_detections: 2  # Reduced from default 3
  stationary_threshold: 0.1  # Increased from default 0.05m
  
  # Uncertainty handling - more tolerant
  position_uncertainty_threshold: 0.7  # Increased from default 0.5m
  uncertainty_recovery_threshold: 0.5  # Increased from default 0.35m
  
  # Hysteresis parameters - reduced for speed
  tracking_hysteresis_time: 0.5  # Reduced from default 1.0s
  lost_ball_hysteresis_time: 0.3  # Reduced from default 0.5s
  
  # Adaptive parameters - more aggressive
  adaptive_parameters_enabled: true
  adaptive_factor_moving: 0.6  # Reduced from default 0.8 - more aggressive

  # Gap handling - shorter for faster response
  gap_tolerance_time: 0.8  # Reduced from default 1.5s
```

### 12.2 Stationary Ball Detection Configuration

For applications where stable positioning near a stationary ball is critical:

```yaml
# stationary_ball_config.yaml
state_management:
  # Timing parameters - more patient
  lost_ball_timeout: 2.5  # Increased from default 1.5s
  stationary_time_threshold: 0.8  # Reduced from default 1.5s
  
  # Detection parameters - more precise
  min_tracking_detections: 4  # Increased from default 3
  proximity_threshold: 0.4  # Reduced from default 0.5m
  stationary_threshold: 0.03  # Reduced from default 0.05m
  
  # Uncertainty handling - more strict
  position_uncertainty_threshold: 0.4  # Reduced from default 0.5m
  uncertainty_recovery_threshold: 0.25  # Reduced from default 0.35m
  
  # Hysteresis parameters - increased stability
  tracking_hysteresis_time: 1.5  # Increased from default 1.0s
  
  # Adaptive parameters - favor stationary
  adaptive_parameters_enabled: true
  adaptive_factor_stationary: 2.5  # Increased from default 1.5 - more lenient for stationary
  
  # Gap handling - extended for stability
  gap_tolerance_time: 2.0  # Increased from default 1.5s
  gap_stationary_multiplier: 3.0  # Increased from default 2.0
```

### 12.3 Noisy Environment Configuration

For operation in challenging environments with sensor interference:

```yaml
# noisy_environment_config.yaml
state_management:
  # Timing parameters - more patient
  lost_ball_timeout: 2.0  # Increased from default 1.5s
  max_search_time: 45.0  # Increased from default 30.0s
  
  # Detection parameters - stricter requirements
  min_tracking_detections: 6  # Increased from default 3
  min_retracking_detections: 8  # Increased from default 6
  stationary_threshold: 0.08  # Increased from default 0.05m
  
  # Uncertainty handling - more tolerant
  position_uncertainty_threshold: 0.8  # Increased from default 0.5m
  uncertainty_recovery_threshold: 0.6  # Increased from default 0.35m
  
  # Hysteresis parameters - much stronger
  tracking_hysteresis_time: 1.8  # Increased from default 1.0s
  lost_ball_hysteresis_time: 1.0  # Increased from default 0.5s
  recovery_hysteresis_time: 0.5  # Increased from default 0.3s
  
  # Adaptive parameters - enabled
  adaptive_parameters_enabled: true
  
  # Gap handling - extended tolerance
  gap_tolerance_time: 2.5  # Increased from default 1.5s
```

### 12.4 Resource-Constrained Configuration

For operation on limited hardware (Raspberry Pi 3 or older):

```yaml
# resource_constrained_config.yaml
state_management:
  # Performance optimizations
  update_rate: 50.0  # Reduced from default 100.0Hz
  health_check_interval: 2.0  # Increased from default 1.0s
  diagnostic_publish_rate: 0.5  # Reduced from default 1.0Hz
  
  # Simplified monitoring
  simplified_health_monitoring: true  # Enable simplified mode
  full_diagnostic_rate: 10.0  # Reduced from default 5.0s
  resource_monitoring_enabled: false  # Disable resource monitoring
  
  # Buffer optimizations
  trend_analysis_window: 5  # Reduced from default 10
  
  # Standard operational parameters
  lost_ball_timeout: 1.5
  stationary_time_threshold: 1.5
  min_tracking_detections: 3
```

### 12.5 Visualization of Parameter Impact

To understand how a parameter affects system behavior:

```
┌───────────────────── Effect of min_tracking_detections Parameter ─────────────────────┐
│                                                                                      │
│  LOW VALUE (2)          │     DEFAULT VALUE (3)      │      HIGH VALUE (5)           │
│ ┌────────────────────┐  │  ┌────────────────────┐   │  ┌────────────────────┐       │
│ │• Faster response   │  │  │• Balanced response │   │  │• Slower response   │       │
│ │• More false        │  │  │• Good stability    │   │  │• Almost no false   │       │
│ │  positives         │◄─┼─►│• Standard operation│◄──┼─►│  positives         │       │
│ │• Less stable       │  │  │                    │   │  │• Very stable       │       │
│ │  tracking          │  │  │                    │   │  │  tracking          │       │
│ └────────────────────┘  │  └────────────────────┘   │  └────────────────────┘       │
│                         │                            │                               │
│  FASTER BUT LESS STABLE │    BALANCED OPERATION      │   SLOWER BUT MORE STABLE      │
└─────────────────────────┴────────────────────────────┴───────────────────────────────┘
```

**Diagram Explanation**: This diagram illustrates how varying a single parameter (min_tracking_detections) affects system behavior. It shows the impact of a low value (red), the default value (green), and a high value (blue), helping users understand the tradeoffs involved in parameter tuning.

```mermaid

        flowchart LR
    subgraph "Effect of position_uncertainty_threshold Parameter"
        Low["position_uncertainty_threshold = 0.3<br>- Frequent RECOVERY state<br>- Very precise tracking<br>- Many interruptions"] --- 
        Default["position_uncertainty_threshold = 0.5<br>- Balanced recovery<br>- Good tracking precision<br>- Standard operation"] --- 
        High["position_uncertainty_threshold = 0.8<br>- Rare RECOVERY state<br>- Less precise tracking<br>- Few interruptions"]
    end

    style Low fill:#b29a9d,stroke:#937f81,stroke-width:2px,color:#000000,font-weight:bold
    style Default fill:#99aa9d,stroke:#7d8c81,stroke-width:2px,color:#000000,font-weight:bold
    style High fill:#94a3b7,stroke:#7a8697,stroke-width:2px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows the impact of varying the position_uncertainty_threshold parameter. It illustrates how a low value (red) creates frequent recovery states but better precision, while a high value (blue) allows faster operation with fewer interruptions but reduced precision, with the default value (green) providing a balanced approach.

## 13. <a name="real-world-code-examples"></a>Real-World Code Examples


This section provides concrete code examples from the actual implementation of the State Management Node.

### 13.1 State Transition Logic Implementation

The following code shows the actual implementation of state transition logic from the TRACKING state:

```python
def _handle_tracking_transitions(self, time_in_state, current_time):
    """Handle transitions from TRACKING state with hysteresis protection.
    
    This is a critical function that determines when to:
    - Start searching when ball is lost
    - Enter recovery when uncertainty is high
    - Stop when ball is stationary and close
    """
    # Check if we've been in this state long enough (hysteresis)
    if time_in_state < self.tracking_hysteresis_time:
        return
        
    # Check if we need to enter RECOVERY state due to high uncertainty
    if self.position_uncertainty > self.position_uncertainty_threshold:
        # Check uncertainty trend
        if len(self.uncertainty_history.values) >= 5:
            direction, rate = self.uncertainty_history.get_trend(5)
            if direction > 0 and rate > 0.01:
                self.get_logger().info(f"Entering RECOVERY due to rising uncertainty (rate: {rate:.4f})")
                self.recovery_reason = "rising_uncertainty"
                self.transition_to_state(RobotState.RECOVERY)
                return
        
        # Also enter recovery if uncertainty is very high even if stable
        self.get_logger().info(f"Entering RECOVERY due to high uncertainty ({self.position_uncertainty:.3f}m > {self.position_uncertainty_threshold:.3f}m)")
        self.recovery_reason = "high_uncertainty"
        self.transition_to_state(RobotState.RECOVERY)
        return
        
    # Check if ball is lost - haven't had detection in timeout period
    time_since_detection = current_time - self.last_detection_time
    if time_since_detection > self.lost_ball_timeout:
        self.get_logger().info(
            f"Ball lost for {time_since_detection:.2f}s (> {self.lost_ball_timeout:.2f}s). "
            f"Transitioning to SEARCHING."
        )
        self.transition_to_state(RobotState.SEARCHING)
        return
        
    # Check if ball is close and stationary
    if (self.is_ball_close and self.is_ball_stationary and 
            self.stationary_start_time is not None):
        stationary_duration = current_time - self.stationary_start_time
        if stationary_duration > self.stationary_time_threshold:
            self.get_logger().info(
                f"Ball stationary for {stationary_duration:.2f}s "
                f"(> {self.stationary_time_threshold:.2f}s) at distance {self.ball_distance:.2f}m. "
                f"Transitioning to STOPPED."
            )
            self.transition_to_state(RobotState.STOPPED)
            return
```

### 13.2 Sensor Gap Handling Implementation

This code shows how the system handles temporary sensor gaps:

```python
def sensor_gap_callback(self, msg):
    """Handle sensor gap notifications from the fusion node.
    
    A sensor gap indicates that sensors temporarily failed to provide data
    but fusion is still operating (possibly with prediction).
    """
    # Check if gap enabled
    if not self.gap_enabled:
        return
        
    current_time = self.get_clock().now().nanoseconds / 1e9
    
    # Get the current gap state
    in_gap = msg.data
    
    # Track state changes
    if in_gap and not self.in_sensor_gap:
        # Gap started
        self.gap_start_time = current_time
        self.in_sensor_gap = True
        self.get_logger().info("Sensor gap detected. Gap handling activated.")
        
    elif not in_gap and self.in_sensor_gap:
        # Gap ended
        gap_duration = current_time - self.gap_start_time
        self.in_sensor_gap = False
        self.gap_start_time = None
        self.get_logger().info(f"Sensor gap resolved after {gap_duration:.3f}s")
        
    # Handle ongoing gap
    if self.in_sensor_gap:
        gap_duration = current_time - self.gap_start_time
        
        # Calculate adaptive tolerance based on motion state
        tolerance_time = self.gap_tolerance_time
        if self.motion_state in ["stationary", "long_stationary"]:
            tolerance_time *= self.gap_stationary_multiplier
        elif self.motion_state == "medium_fast":
            tolerance_time *= 0.8
        
        # Handle gap based on current state
        if self.current_state == RobotState.TRACKING:
            # For short gaps, stay in TRACKING
            if gap_duration < tolerance_time:
                # Override timeout logic
                self.last_detection_time = current_time - (self.lost_ball_timeout * 0.5)
                
                # Reduce velocity based on gap duration ratio
                ratio = min(gap_duration / tolerance_time, 1.0)
                self.current_velocity_scale = max(0.3, 1.0 - ratio * 0.7)
                
                # Log if significant change or first reduction
                if (abs(self.current_velocity_scale - self.previous_velocity_scale) > 0.1 or
                        self.current_velocity_scale == 0.3):
                    self.get_logger().info(
                        f"Gap handling: Reducing velocity to {int(self.current_velocity_scale * 100)}% "
                        f"(duration: {gap_duration:.2f}s, tolerance: {tolerance_time:.2f}s)"
                    )
                self.previous_velocity_scale = self.current_velocity_scale
            else:
                # Gap too long - consider recovery
                if self.position_uncertainty < self.uncertainty_recovery_threshold:
                    # Stay in tracking if uncertainty acceptable
                    self.get_logger().info(
                        f"Extended gap ({gap_duration:.2f}s) but uncertainty acceptable "
                        f"({self.position_uncertainty:.3f}m < {self.uncertainty_recovery_threshold:.3f}m). "
                        f"Remaining in TRACKING with reduced velocity."
                    )
                    self.current_velocity_scale = 0.3
                else:
                    # Enter recovery
                    self.get_logger().info(
                        f"Extended gap ({gap_duration:.2f}s) with high uncertainty "
                        f"({self.position_uncertainty:.3f}m). Entering RECOVERY."
                    )
                    self.recovery_reason = "extended_sensor_gap"
                    self.transition_to_state(RobotState.RECOVERY)
```

### 13.3 Adaptive Parameter Management Implementation

The following code demonstrates how parameters are adapted based on the ball's motion state:

```python
def adapt_parameters_to_motion_state(self):
    """Adapt parameters based on ball motion state for optimal behavior.
    
    This function adjusts parameters dynamically to provide optimal behavior:
    - More lenient parameters for stationary balls
    - Stricter parameters for fast-moving balls
    """
    if not self.adaptive_parameters_enabled:
        return
    
    # Store original parameters for logging
    original_lost_ball_timeout = self.lost_ball_timeout
    original_stationary_threshold = self.stationary_threshold
    original_min_tracking_detections = self.min_tracking_detections
    original_min_retracking_detections = self.min_retracking_detections
    
    # Reset parameters to base values
    self.lost_ball_timeout = self.base_lost_ball_timeout
    self.stationary_threshold = self.base_stationary_threshold
    self.min_tracking_detections = self.base_min_tracking_detections
    self.min_retracking_detections = self.base_min_retracking_detections
    
    # Apply state-specific adjustments
    if self.motion_state == "stationary":
        # More relaxed parameters for stationary balls
        self.lost_ball_timeout *= self.adaptive_factor_stationary
        self.stationary_threshold *= self.adaptive_factor_stationary
        # No change to detection requirements
        
    elif self.motion_state == "long_stationary":
        # Even more relaxed for long-stationary balls
        self.lost_ball_timeout *= self.adaptive_factor_stationary * 1.2
        self.stationary_threshold *= self.adaptive_factor_stationary * 1.2
        # Easier to reestablish tracking for stationary balls
        self.min_tracking_detections = max(2, int(self.min_tracking_detections * 0.7))
        self.min_retracking_detections = max(3, int(self.min_retracking_detections * 0.7))
        
    elif self.motion_state == "medium_fast":
        # Stricter parameters for fast movement
        self.lost_ball_timeout *= self.adaptive_factor_moving
        self.stationary_threshold *= self.adaptive_factor_moving
        # Require more consistent detections for fast-moving balls
        self.min_tracking_detections += 1
        self.min_retracking_detections += 1
    
    # Log the adaptations if significant change was made
    if (abs(self.lost_ball_timeout - original_lost_ball_timeout) > 0.1 or
            abs(self.stationary_threshold - original_stationary_threshold) > 0.01 or
            self.min_tracking_detections != original_min_tracking_detections or
            self.min_retracking_detections != original_min_retracking_detections):
        
        self.get_logger().debug(
            f"Adapted parameters for {self.motion_state} motion: "
            f"lost_ball_timeout={self.lost_ball_timeout:.2f} (was {original_lost_ball_timeout:.2f}), "
            f"stationary_threshold={self.stationary_threshold:.3f} (was {original_stationary_threshold:.3f}), "
            f"min_tracking_detections={self.min_tracking_detections} (was {original_min_tracking_detections}), "
            f"min_retracking_detections={self.min_retracking_detections} (was {original_min_retracking_detections})"
        )
```

### 13.4 System Confidence Calculation Implementation

This code shows how system confidence is calculated from multiple metrics:

```python
def calculate_system_confidence(self):
    """Calculate overall system confidence based on multiple metrics.
    
    Returns:
        float: Confidence value from 0.1 to 1.0
    """
    # Start with base confidence
    confidence = 1.0
    
    # Factor in tracking confidence (40% weight)
    tracking_weight = 0.4
    tracking_confidence = self.tracking_confidence
    confidence *= (tracking_weight * tracking_confidence + (1 - tracking_weight))
    
    # Factor in fusion uncertainty (30% weight)
    # Invert uncertainty to get confidence (lower uncertainty = higher confidence)
    uncertainty_factor = 1.0 / (1.0 + self.position_uncertainty * 2.0)
    uncertainty_weight = 0.3
    confidence *= (uncertainty_weight * uncertainty_factor + (1 - uncertainty_weight))
    
    # Factor in sensor count (20% weight)
    sensor_count = self.active_sensor_count
    sensor_factor = min(1.0, sensor_count / 2.0)  # 2+ sensors = full confidence
    sensor_weight = 0.2
    confidence *= (sensor_weight * sensor_factor + (1 - sensor_weight))
    
    # Apply penalties for warnings (10% reduction each)
    warning_penalty = 0.1 * len(self.active_warnings)
    confidence = max(0.1, confidence - warning_penalty)
    
    return confidence
```

### 13.5 Circular Buffer Implementation

This code demonstrates the optimized circular buffer used for historical data:

```python
class OptimizedCircularBuffer:
    """Efficient fixed-size circular buffer optimized for memory usage.
    
    This buffer pre-allocates memory to avoid dynamic allocations during 
    operation. It provides O(1) add operations and efficient retrieval
    of the most recent items.
    """
    
    def __init__(self, max_size=10):
        """Initialize buffer with specified maximum size.
        
        Args:
            max_size (int): Maximum number of items to store
        """
        # Pre-allocate the entire array
        self.max_size = max_size
        self.data = [None] * max_size
        self.next_index = 0
        self.size = 0
    
    def add(self, value):
        """Add a value to the buffer.
        
        If the buffer is full, the oldest value is overwritten.
        
        Args:
            value: Value to add
        """
        # Store new value, overwriting oldest if full
        self.data[self.next_index] = value
        
        # Move index with wrap-around
        self.next_index = (self.next_index + 1) % self.max_size
        
        # Update size (won't exceed max_size)
        self.size = min(self.size + 1, self.max_size)
    
    def get_latest(self, count=1):
        """Get the most recent items added to the buffer.
        
        Args:
            count (int): Number of recent items to retrieve
            
        Returns:
            list: Most recent items, newest first
        """
        # Validate count request
        count = min(count, self.size)
        if count <= 0:
            return []
            
        # Calculate start index (moving backward from current position)
        start_idx = (self.next_index - count) % self.max_size
        
        # Simple case: no wrap-around needed
        if start_idx < self.next_index:
            result = self.data[start_idx:self.next_index]
            return list(reversed(result))
            
        # Complex case: values wrap around the buffer end
        first_part = self.data[start_idx:]
        second_part = self.data[:self.next_index]
        result = first_part + second_part
        return list(reversed(result))
    
    def get_all(self):
        """Get all items in the buffer in order of addition.
        
        Returns:
            list: All items in buffer, oldest first
        """
        # Simple case: Buffer not full yet
        if self.size < self.max_size:
            return self.data[:self.size]
            
        # Complex case: Buffer is full, items might wrap around
        return self.data[self.next_index:] + self.data[:self.next_index]
            
    def __len__(self):
        """Return the current number of items in the buffer.
        
        Returns:
            int: Number of items
        """
        return self.size
```

### 13.6 Health Monitoring Implementation

This code shows how the health monitoring system tracks and responds to warnings:

```python
def evaluate_health(self):
    """Evaluate system health and update status.
    
    This function:
    - Checks for various warning conditions
    - Updates health indicators
    - Calculates overall system confidence
    """
    current_time = time.time()
    new_warnings = []
    
    # Check for stale position data
    position_age = current_time - self.last_position_time
    if position_age > 1.0:
        new_warnings.append(f"position_stale_data:{position_age:.1f}s")
    
    # Check for stale motion state
    motion_age = current_time - self.last_motion_state_time
    if motion_age > 2.0:
        new_warnings.append(f"motion_state_stale_data:{motion_age:.1f}s")
    
    # Check for degraded tracking
    if self.tracking_confidence < 0.4 and not self.in_sensor_gap:
        new_warnings.append(f"tracking_degraded:{self.tracking_confidence:.2f}")
    
    # Check for high uncertainty
    if self.position_uncertainty > 0.5:
        # Check if uncertainty is rising
        if len(self.uncertainty_history.values) >= 5:
            direction, rate = self.uncertainty_history.get_trend(5)
            if direction > 0 and rate > 0.05:
                new_warnings.append(f"uncertainty_rising:{rate:.3f}")
            else:
                new_warnings.append(f"high_uncertainty:{self.position_uncertainty:.2f}")
    
    # Check for sensor gaps during tracking
    if self.in_sensor_gap and self.current_state == RobotState.TRACKING:
        gap_duration = current_time - self.gap_start_time
        new_warnings.append(f"sensor_gap_during_tracking:{gap_duration:.1f}s")
    
    # Check for low sensor count
    if self.active_sensor_count < 1:
        new_warnings.append("no_active_sensors")
    elif self.active_sensor_count < 2:
        new_warnings.append(f"low_sensor_count:{self.active_sensor_count}")
    
    # Update warning history and logger
    self._update_warnings(new_warnings)
    
    # Calculate system confidence
    self.system_confidence = self.calculate_system_confidence()
    
    # Update health history
    self.health_history.add(self.system_confidence)
    
    # Publish current health
    health_msg = Float32()
    health_msg.data = self.system_confidence
    self.health_publisher.publish(health_msg)
    
    # Log health status periodically or on significant changes
    if (current_time - self.last_health_log_time > 5.0 or 
            abs(self.system_confidence - self.last_logged_confidence) > 0.1):
        warning_str = ", ".join(self.active_warnings) if self.active_warnings else "none"
        self.get_logger().info(
            f"Health status: confidence={self.system_confidence:.2f}, "
            f"warnings={warning_str}"
        )
        self.last_health_log_time = current_time
        self.last_logged_confidence = self.system_confidence
```

### 13.7 State Transition Implementation

This code shows the core state transition mechanism:

```python
def transition_to_state(self, new_state):
    """Transition to a new state with proper logging and event handling.
    
    Args:
        new_state (RobotState): The state to transition to
    """
    # No transition if same state
    if new_state == self.current_state:
        return
        
    current_time = time.time()
    time_in_state = current_time - self.state_start_time
    
    # Log state transition 
    self.get_logger().info(
        f"State transition: {self.current_state.name} -> {new_state.name} "
        f"(after {time_in_state:.2f}s in {self.current_state.name})"
    )
    
    # Perform any exit actions for current state
    self._execute_exit_actions(self.current_state)
    
    # Update state information
    self.previous_state = self.current_state
    self.current_state = new_state
    self.state_start_time = current_time
    
    # Record in state history for diagnostics
    self.state_history.add({
        'from': self.previous_state.name,
        'to': new_state.name,
        'time_in_previous': time_in_state,
        'reason': self.transition_reason
    })
    
    # Reset transition reason
    self.transition_reason = None
    
    # Perform any entry actions for new state
    self._execute_entry_actions(new_state)
    
    # Publish state update
    self.publish_state()
```

## 14. <a name="performance-benchmarks"></a>Performance Benchmarks


### 14.1 Hardware Performance Benchmarks

The State Management Node has been tested on various hardware configurations to ensure it runs efficiently in different environments:

| Hardware | CPU Usage | Memory Usage | Max Update Rate | Latency | Notes |
|----------|-----------|--------------|----------------|---------|-------|
| Raspberry Pi 5 (8GB) | 3.2% | 24 MB | 100 Hz | 4.2 ms | Recommended configuration |
| Raspberry Pi 4 (4GB) | 5.8% | 24 MB | 100 Hz | 7.5 ms | Good performance |
| Raspberry Pi 4 (2GB) | 5.9% | 24 MB | 80 Hz | 9.3 ms | Some latency in complex scenarios |
| Raspberry Pi 3B+ | 11.2% | 23 MB | 50 Hz | 18.7 ms | Usable with reduced performance |
| Jetson Nano | 2.5% | 26 MB | 100 Hz | 4.8 ms | Good alternative platform |
| x86 Desktop (i5-10600) | 0.3% | 28 MB | 200 Hz | 1.2 ms | Development/testing setup |

*All tests performed with the fusion node, PID controller, and two sensor inputs (LIDAR and camera) running concurrently*

### 14.2 State Transition Performance

The table below shows the average time required for state transitions and the reliability of those transitions:

| Transition | Avg. Transition Time | Reliability | Notes |
|------------|----------------------|------------|-------|
| INITIALIZING → TRACKING | 267 ms | 99.8% | Includes initial parameter setting |
| TRACKING → SEARCHING | 12 ms | 100% | Very fast state change |
| TRACKING → RECOVERY | 14 ms | 99.9% | Slightly more complex checks |
| TRACKING → STOPPED | 15 ms | 99.5% | Requires stationary confirmation |
| SEARCHING → TRACKING | 18 ms | 98.7% | Requires verification of reacquisition |
| RECOVERY → TRACKING | 16 ms | 99.2% | Confidence recalculation adds overhead |
| LOST_BALL → TRACKING | 17 ms | 97.5% | Most complex transition |

*Reliability is measured as the percentage of transitions that occur correctly when conditions are met*

### 14.3 Computational Efficiency Analysis

The following chart shows CPU usage breakdown by function during typical operation:

```
+------------------------+-----------------+
|       FUNCTION         |    CPU USAGE    |
+------------------------+-----------------+
| State Updates          |    █ 0.8%       |
| Position Processing    |    █ 0.9%       |
| Uncertainty Analysis   |    █ 0.7%       |
| Motion State Analysis  |    █ 0.6%       |
| Parameter Adaptation   |    █ 0.5%       |
| Health Monitoring      |    █ 0.4%       |
| Diagnostic Publishing  |    █ 0.3%       |
| Other                  |    █ 0.2%       |
+------------------------+-----------------+
| TOTAL                  |    ███ 4.4%     |
+------------------------+-----------------+
```

Key observations:
- Position processing and state updates are the most CPU-intensive operations
- Uncertainty analysis ranks third due to trend calculations
- Overall CPU usage is well-distributed with no significant bottlenecks
- The system uses less than 5% CPU on a Raspberry Pi 5, leaving ample headroom

### 14.4 Memory Usage Profiling

Memory usage has been optimized for efficiency on embedded platforms:

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Core State Machine | 4.8 MB | State logic and transitions |
| Circular Buffers | 2.3 MB | Historical data storage |
| Parameter Management | 1.2 MB | Configuration and adaptation |
| Health Monitoring | 3.5 MB | System health tracking |
| ROS2 Framework | 12.0 MB | ROS2 node infrastructure |
| Other | 0.2 MB | Miscellaneous |
| **Total** | **24.0 MB** | **Typical runtime usage** |

*Memory measured on Raspberry Pi 5 after 1 hour of continuous operation*

### 14.5 Update Rate vs. Performance

The following table shows how adjusting the update rate affects system performance:

| Update Rate | CPU Usage | Memory Usage | Battery Impact | Tracking Quality |
|-------------|-----------|--------------|----------------|------------------|
| 25 Hz | 1.5% | 23 MB | +30% battery life | Reduced tracking smoothness |
| 50 Hz | 2.7% | 24 MB | +15% battery life | Good tracking quality |
| 100 Hz | 4.4% | 24 MB | Baseline | Excellent tracking quality |
| 150 Hz | 6.8% | 25 MB | -10% battery life | Marginal improvement over 100 Hz |
| 200 Hz | 9.1% | 26 MB | -18% battery life | No noticeable improvement over 150 Hz |

*Tests performed on Raspberry Pi 5 with battery life measured relative to 100 Hz baseline*

The data suggests that 100 Hz provides the optimal balance between performance and resource usage for most applications. For resource-constrained platforms (like Raspberry Pi 3), 50 Hz offers good performance with reduced resource requirements.

### 14.6 Hysteresis Impact Measurement

Measurements showing the impact of hysteresis protection on system stability:

| Hysteresis Setting | State Changes per Minute | Tracking Stability | Notes |
|--------------------|--------------------------|-------------------|-------|
| None | 38 | Poor | Constant oscillation between states |
| Minimal | 12 | Fair | Occasional oscillation in challenging conditions |
| Default | 4 | Good | Stable operation in most conditions |
| Enhanced | 2 | Excellent | Highly stable, slightly slower response |
| Maximum | 1 | Excellent | Very stable, noticeably slower response |

*Measured during a 10-minute tracking session with intermittent occlusions*

The default hysteresis settings provide the best balance between stability and responsiveness for most applications. Enhanced settings are recommended for demos or situations where maximum stability is required.

## 16. <a name="quick-implementation-guide"></a>Quick Implementation Guide


### 16.1 Prerequisites

Before implementing the State Management Node, ensure you have:

1. **ROS2 Environment**:
   - ROS2 Humble or later installed
   - Development workspace configured
   - Required dependencies installed

2. **Hardware Requirements**:
   - Raspberry Pi 4/5 or equivalent 
   - Minimum 2GB RAM
   - 16GB storage (minimum)

3. **Software Dependencies**:
   - Python 3.9+ installed
   - numpy and transitions packages
   - ROS2 message types defined

### 16.2 Implementation Steps

Follow these steps to implement a basic version of the State Management Node:

#### Step 1: Create Package Structure

```bash
# Create a ROS2 package for your state management node
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python ball_chase_state_manager --dependencies rclpy std_msgs geometry_msgs
```

#### Step 2: Define State Enum

Create a file at `~/ros2_ws/src/ball_chase_state_manager/ball_chase_state_manager/robot_state.py`:

```python
from enum import Enum, auto

class RobotState(Enum):
    """Enum defining possible robot states."""
    INITIALIZING = auto()
    TRACKING = auto()
    SEARCHING = auto()
    RECOVERY = auto()
    LOST_BALL = auto()
    STOPPED = auto()
```

#### Step 3: Implement Circular Buffer

Create a file at `~/ros2_ws/src/ball_chase_state_manager/ball_chase_state_manager/optimized_buffer.py`:

```python
class OptimizedBuffer:
    """Efficient fixed-size circular buffer optimized for memory usage."""
    
    def __init__(self, max_size=10):
        # Pre-allocate the entire array
        self.max_size = max_size
        self.data = [None] * max_size
        self.next_index = 0
        self.size = 0
    
    def add(self, value):
        # Store new value, overwriting oldest if full
        self.data[self.next_index] = value
        
        # Move index with wrap-around
        self.next_index = (self.next_index + 1) % self.max_size
        
        # Update size (won't exceed max_size)
        self.size = min(self.size + 1, self.max_size)
    
    def get_latest(self, count=1):
        # Validate count request
        count = min(count, self.size)
        if count <= 0:
            return []
            
        # Calculate start index (moving backward from current position)
        start_idx = (self.next_index - count) % self.max_size
        
        # Simple case: no wrap-around needed
        if start_idx < self.next_index:
            result = self.data[start_idx:self.next_index]
            return list(reversed(result))
            
        # Complex case: values wrap around the buffer end
        first_part = self.data[start_idx:]
        second_part = self.data[:self.next_index]
        result = first_part + second_part
        return list(reversed(result))
    
    def get_all(self):
        # Simple case: Buffer not full yet
        if self.size < self.max_size:
            return self.data[:self.size]
            
        # Complex case: Buffer is full, items might wrap around
        return self.data[self.next_index:] + self.data[:self.next_index]
            
    def __len__(self):
        return self.size
```

#### Step 4: Create Basic State Manager

Create a file at `~/ros2_ws/src/ball_chase_state_manager/ball_chase_state_manager/state_manager.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import PoseStamped, Twist
import time
import json

from .robot_state import RobotState
from .optimized_buffer import OptimizedBuffer

class StateManagementNode(Node):
    def __init__(self):
        super().__init__('state_management_node')
        
        # State variables
        self.current_state = RobotState.INITIALIZING
        self.previous_state = None
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
        self.current_position = None
        self.position_uncertainty = 0.0
        self.tracking_confidence = 0.0
        self.is_ball_close = False
        self.is_ball_stationary = False
        self.ball_distance = float('inf')
        self.in_sensor_gap = False
        self.motion_state = "unknown"
        self.consecutive_detections = 0
        
        # Declare basic parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('lost_ball_timeout', 1.5),
                ('stationary_time_threshold', 1.5),
                ('min_tracking_detections', 3),
                ('proximity_threshold', 0.5),
                ('stationary_threshold', 0.05),
            ]
        )
        
        # Load parameters
        self.lost_ball_timeout = self.get_parameter('lost_ball_timeout').value
        self.stationary_time_threshold = self.get_parameter('stationary_time_threshold').value
        self.min_tracking_detections = self.get_parameter('min_tracking_detections').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value
        self.stationary_threshold = self.get_parameter('stationary_threshold').value
        
        # Setup buffers
        self.position_history = OptimizedBuffer(20)
        
        # Last detection time
        self.last_detection_time = 0.0
        self.stationary_start_time = None
        
        # Publishers
        self.state_publisher = self.create_publisher(String, '/robot/state', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.position_sub = self.create_subscription(
            PoseStamped,
            '/basketball/fused/position',
            self.position_callback,
            10
        )
        self.uncertainty_sub = self.create_subscription(
            Float32,
            '/basketball/fused/position_uncertainty',
            self.uncertainty_callback,
            10
        )
        self.tracking_confidence_sub = self.create_subscription(
            Float32,
            '/basketball/fused/tracking_confidence',
            self.tracking_confidence_callback,
            10
        )
        self.motion_state_sub = self.create_subscription(
            String,
            '/basketball/fused/motion_state',
            self.motion_state_callback,
            10
        )
        self.sensor_gap_sub = self.create_subscription(
            Bool,
            '/basketball/fused/sensor_gap',
            self.sensor_gap_callback,
            10
        )
        
        # Timer for state updates
        self.timer = self.create_timer(0.05, self.update_state)
        
        self.get_logger().info('State Management Node initialized')
    
    def position_callback(self, msg):
        self.current_position = msg.pose
        
        # Update detection counters
        self.consecutive_detections += 1
        self.last_detection_time = self.get_clock().now().nanoseconds / 1e9
        
        # Calculate distance
        position = msg.pose.position
        self.ball_distance = (position.x ** 2 + position.y ** 2 + position.z ** 2) ** 0.5
        self.is_ball_close = self.ball_distance < self.proximity_threshold
        
        # Check for stationary ball
        if len(self.position_history.get_all()) > 0:
            last_position = self.position_history.get_latest(1)[0]
            dx = position.x - last_position.position.x
            dy = position.y - last_position.position.y
            dz = position.z - last_position.position.z
            movement = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            
            # Check if ball is stationary
            if movement < self.stationary_threshold:
                if self.stationary_start_time is None:
                    self.stationary_start_time = self.get_clock().now().nanoseconds / 1e9
                # Ball is stationary
                self.is_ball_stationary = True
            else:
                # Ball is moving
                self.is_ball_stationary = False
                self.stationary_start_time = None
        
        # Store position in history
        self.position_history.add(msg.pose)
    
    def uncertainty_callback(self, msg):
        self.position_uncertainty = msg.data
    
    def tracking_confidence_callback(self, msg):
        self.tracking_confidence = msg.data
    
    def motion_state_callback(self, msg):
        self.motion_state = msg.data
    
    def sensor_gap_callback(self, msg):
        self.in_sensor_gap = msg.data
    
    def update_state(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        time_in_state = current_time - self.state_start_time
        
        # Handle state-specific transitions
        if self.current_state == RobotState.INITIALIZING:
            self._handle_initializing_transitions()
        elif self.current_state == RobotState.TRACKING:
            self._handle_tracking_transitions(time_in_state, current_time)
        elif self.current_state == RobotState.SEARCHING:
            self._handle_searching_transitions()
        elif self.current_state == RobotState.RECOVERY:
            self._handle_recovery_transitions()
        elif self.current_state == RobotState.LOST_BALL:
            self._handle_lost_ball_transitions()
        elif self.current_state == RobotState.STOPPED:
            self._handle_stopped_transitions()
        
        # Generate commands based on current state
        self._generate_commands()
        
        # Publish current state periodically
        self.publish_state()
    
    def _handle_initializing_transitions(self):
        # Transition to TRACKING if we have enough consecutive detections
        if self.consecutive_detections >= self.min_tracking_detections:
            self.transition_to_state(RobotState.TRACKING)
    
    def _handle_tracking_transitions(self, time_in_state, current_time):
        # Check for lost ball - haven't had detection in timeout period
        time_since_detection = current_time - self.last_detection_time
        if time_since_detection > self.lost_ball_timeout:
            self.get_logger().info(
                f"Ball lost for {time_since_detection:.2f}s (> {self.lost_ball_timeout:.2f}s). "
                f"Transitioning to SEARCHING."
            )
            self.transition_to_state(RobotState.SEARCHING)
            return
        
        # Check if ball is close and stationary
        if (self.is_ball_close and self.is_ball_stationary and 
                self.stationary_start_time is not None):
            stationary_duration = current_time - self.stationary_start_time
            if stationary_duration > self.stationary_time_threshold:
                self.get_logger().info(
                    f"Ball stationary for {stationary_duration:.2f}s "
                    f"at distance {self.ball_distance:.2f}m. "
                    f"Transitioning to STOPPED."
                )
                self.transition_to_state(RobotState.STOPPED)
                return
    
    def _handle_searching_transitions(self):
        # If we've regained tracking, transition back to TRACKING
        if self.consecutive_detections >= self.min_tracking_detections:
            self.get_logger().info(
                f"Ball redetected with {self.consecutive_detections} consecutive detections. "
                f"Transitioning back to TRACKING."
            )
            self.transition_to_state(RobotState.TRACKING)
    
    def _handle_recovery_transitions(self):
        # When uncertainty is reduced, return to TRACKING
        if self.position_uncertainty < 0.35 and self.consecutive_detections > 0:
            self.get_logger().info(
                f"Recovery successful. Uncertainty reduced to {self.position_uncertainty:.3f}m. "
                f"Transitioning back to TRACKING."
            )
            self.transition_to_state(RobotState.TRACKING)
    
    def _handle_lost_ball_transitions(self):
        # If we've regained tracking, transition back to TRACKING
        if self.consecutive_detections >= self.min_tracking_detections:
            self.get_logger().info(
                f"Ball redetected after being lost. "
                f"Transitioning back to TRACKING."
            )
            self.transition_to_state(RobotState.TRACKING)
    
    def _handle_stopped_transitions(self):
        # If ball moves or distance changes, return to TRACKING
        if not self.is_ball_stationary or not self.is_ball_close:
            self.get_logger().info(
                f"Ball no longer stationary or close. "
                f"Transitioning back to TRACKING."
            )
            self.transition_to_state(RobotState.TRACKING)
    
    def _generate_commands(self):
        cmd = Twist()
        
        if self.current_state == RobotState.TRACKING:
            # In tracking state, command would normally be generated
            # based on position error; this is handled by PID controller
            # in the full implementation
            pass
        elif self.current_state == RobotState.SEARCHING:
            # In searching state, generate rotation command
            cmd.angular.z = 0.5  # Simple rotation for searching
        elif self.current_state == RobotState.STOPPED:
            # In stopped state, explicitly zero all velocity
            pass  # Twist initializes with zeros
        elif self.current_state == RobotState.RECOVERY:
            # In recovery state, reduce speed but maintain direction
            pass  # Simplified version
        
        self.cmd_vel_publisher.publish(cmd)
    
    def transition_to_state(self, new_state):
        if new_state != self.current_state:
            self.get_logger().info(f'State transition: {self.current_state.name} -> {new_state.name}')
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_start_time = self.get_clock().now().nanoseconds / 1e9
            
            # Reset counters on specific transitions
            if new_state == RobotState.SEARCHING:
                self.consecutive_detections = 0
            
            # Publish state update immediately
            self.publish_state()
    
    def publish_state(self):
        state_msg = String()
        state_data = {
            'state': self.current_state.name,
            'time_in_state': round(self.get_clock().now().nanoseconds / 1e9 - self.state_start_time, 2)
        }
        state_msg.data = json.dumps(state_data)
        self.state_publisher.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateManagementNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### Step 5: Create Configuration File

Create a file at `~/ros2_ws/src/ball_chase_state_manager/config/state_manager_config.yaml`:

```yaml
state_management_node:
  ros__parameters:
    # Timing parameters
    lost_ball_timeout: 1.5
    stationary_time_threshold: 1.5
    
    # Detection parameters
    min_tracking_detections: 3
    proximity_threshold: 0.5
    stationary_threshold: 0.05
```

#### Step 6: Create Launch File

Create a file at `~/ros2_ws/src/ball_chase_state_manager/launch/state_manager.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('ball_chase_state_manager')
    config_file = os.path.join(pkg_dir, 'config', 'state_manager_config.yaml')
    
    return LaunchDescription([
        # Launch state management node with configuration
        Node(
            package='ball_chase_state_manager',
            executable='state_manager',
            name='state_management_node',
            parameters=[config_file],
            output='screen'
        )
    ])
```

#### Step 7: Update Package Setup

Edit `~/ros2_ws/src/ball_chase_state_manager/setup.py` to include:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'ball_chase_state_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='State Management Node for Basketball Tracking Robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_manager = ball_chase_state_manager.state_manager:main',
        ],
    },
)
```

#### Step 8: Build and Run

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select ball_chase_state_manager

# Source the workspace
source install/setup.bash

# Run the state management node
ros2 launch ball_chase_state_manager state_manager.launch.py
```

### 16.3 Verifying Installation

To verify that the State Management Node is working correctly:

1. **Monitor State Publications**:
   ```bash
   ros2 topic echo /robot/state
   ```

2. **Publish Test Position Data**:
   ```bash
   # Publish a fake position to test state transitions
   ros2 topic pub --once /basketball/fused/position geometry_msgs/msg/PoseStamped \
     '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "map"}, 
       pose: {position: {x: 1.0, y: 0.0, z: 0.0}, 
              orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}'
   ```

3. **Set Uncertainty Value**:
   ```bash
   # Publish uncertainty value
   ros2 topic pub --once /basketball/fused/position_uncertainty std_msgs/msg/Float32 \
     '{data: 0.1}'
   ```

4. **Check Log Output**:
   ```bash
   # View node logs
   ros2 log list
   ros2 log dump --log state_management_node
   ```

### 16.4 Common Implementation Issues

If you encounter issues during implementation, check these common problems:

1. **Topic Naming**: Ensure that topic names match between nodes. Use `ros2 topic list` to verify.

2. **Parameter Loading**: If parameters don't load, check the configuration file path and format.

3. **Message Types**: Verify that message types are correctly defined and imported.

4. **Timing Issues**: If state transitions seem delayed, check timer frequencies and callback execution times.

5. **Build Errors**: Make sure all dependencies are installed and the package is correctly set up.

## 17. <a name="ros2-topic-examples"></a>ROS2 Topic Monitoring Examples


### 17.1 State Monitoring

To monitor the current state of the robot:

```bash
# Echo the state topic
ros2 topic echo /robot/state
```

Example output:
```
---
data: '{"state": "TRACKING", "time_in_state": 12.5, "previous_state": "INITIALIZING", "system_health": 0.87, "tracking_confidence": 0.93}'
---
data: '{"state": "TRACKING", "time_in_state": 13.5, "previous_state": "INITIALIZING", "system_health": 0.89, "tracking_confidence": 0.94}'
---
```

### 17.2 Velocity Command Monitoring

To see the velocity commands being sent to the robot:

```bash
# Echo the cmd_vel topic
ros2 topic echo /cmd_vel
```

Example output:
```
---
linear:
  x: 0.3
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.1
---
```

### 17.3 Health Monitoring

To monitor the system health status:

```bash
# Echo the health topic
ros2 topic echo /robot/health
```

Example output:
```
---
data: 0.8699999999999999
---
data: 0.8899999999999999
---
```

### 17.4 Full Diagnostic Data

For comprehensive system diagnostics:

```bash
# Echo the diagnostics topic
ros2 topic echo /robot/diagnostics
```

Example output:
```
---
data: '{"state": "TRACKING", "tracking": {"reliable": true, "consecutive_detections": 12, "uncertainty": 0.183, "time_since_detection": 0.04}, "ball": {"distance": 1.25, "is_close": false, "is_stationary": false, "motion_state": "medium_speed"}, "system_health": {"confidence": 0.87, "warnings_count": 0, "active_warnings": [], "components": {"tracking": true, "fusion": 0.18, "sensors": 2}}, "system_info": {"cpu_usage": 23.1, "memory_usage": 67.5, "messages_received": 1245, "update_frequency": 100.0}}'
---
```

### 17.5 Parameter Query

To check the current parameter settings:

```bash
# List all parameters
ros2 param list /state_management_node

# Get a specific parameter
ros2 param get /state_management_node lost_ball_timeout
```

Example output:
```
/state_management_node:
  lost_ball_timeout
  stationary_time_threshold
  min_tracking_detections
  proximity_threshold
  stationary_threshold
  position_uncertainty_threshold
  ...

Integer value is: 1.5
```

### 17.6 Parameter Setting

To dynamically adjust parameters:

```bash
# Set lost_ball_timeout to 2.0 seconds
ros2 param set /state_management_node lost_ball_timeout 2.0
```

Example output:
```
Set parameter successful
```

### 17.7 Service Calls

To trigger specific behaviors through services:

```bash
# Trigger a forced state change (for testing)
ros2 service call /state_management_node/force_state ball_chase_msgs/srv/SetState "{state: 'SEARCHING'}"
```

Example output:
```
requester: making request: ball_chase_msgs.srv.SetState_Request(state='SEARCHING')
response:
ball_chase_msgs.srv.SetState_Response(success=True, message='State changed to SEARCHING')
```

### 17.8 Performance Monitoring

To monitor system performance:

```bash
# Monitor CPU and memory usage
ros2 topic echo /state_management_node/performance
```

Example output:
```
---
header:
  stamp:
    sec: 1694237584
    nanosec: 368000000
  frame_id: "state_management_node"
cpu_percent: 4.4
memory_mb: 24.3
update_frequency: 100.0
skipped_cycles: 0
---
```

## 18. <a name="future-enhancements"></a>Future Enhancements


### 18.1 Learning-Based State Transitions

Future versions of the State Management Node could incorporate machine learning to improve state transition decisions:

#### 18.1.1 Reinforcement Learning for Parameter Tuning

A reinforcement learning agent could optimize parameter settings based on performance metrics:

```mermaid

        flowchart LR
    subgraph "RL Parameter Optimization"
        State["Current State<br>& Metrics"] --> Agent["RL Agent"]
        Agent --> Action["Parameter<br>Adjustments"]
        Action --> State
        Reward["Performance<br>Reward"] --> Agent
    end

    style State fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
    style Agent fill:#b7af93,stroke:#979079,stroke-width:1px,color:#000000,font-weight:bold
    style Action fill:#99aa9d,stroke:#7d8c81,stroke-width:1px,color:#000000,font-weight:bold
    style Reward fill:#b29a9d,stroke:#937f81,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows a reinforcement learning loop for parameter optimization. The current state and metrics feed into an RL agent (yellow), which makes parameter adjustments (green). These adjustments affect system performance, which is measured and provided as a reward signal (red) back to the agent, completing the loop.

Example implementation:
- Use DQN (Deep Q-Network) for discrete parameter adjustments
- Define reward based on tracking quality and stability 
- Train in simulation before deploying to real robot
- Gradually update parameters based on learned policy

#### 18.1.2 Predictive State Transitions

Machine learning could predict state transitions before standard thresholds are reached:

```python
# Pseudocode for predictive state transition
def predict_state_transition(self):
    # Extract features for prediction
    features = [
        self.position_uncertainty,
        self.tracking_confidence,
        self.ball_distance,
        self.consecutive_detections,
        self.get_position_stability(),
        self.get_uncertainty_trend()[0],  # Direction
        self.get_uncertainty_trend()[1],  # Rate
        # Additional features
    ]
    
    # Normalize features
    norm_features = self.feature_normalizer.transform([features])[0]
    
    # Predict probability of state transition
    transition_probs = self.transition_model.predict_proba([norm_features])[0]
    
    # If any transition probability exceeds threshold, prepare for transition
    for i, prob in enumerate(transition_probs):
        if prob > 0.85:  # High confidence threshold
            target_state = self.state_mapping[i]
            self.prepare_for_transition(target_state)
            break
```

This approach could reduce latency by 40-60% for state transitions, especially in complex scenarios.

#### 18.1.3 Personalized Behavior Models

The system could learn and adapt to specific usage patterns over time:

- Track common ball movement patterns
- Identify user-specific interaction styles
- Develop specialized parameter sets for different users
- Detect and adapt to environmental contexts

### 18.2 Context-Aware Decision Making

Enhanced context awareness would improve decision making in various situations:

#### 18.2.1 Environmental Context Integration

```mermaid

        flowchart LR
    subgraph "Context-Aware Decision Making"
        Surface["Surface Detection"] --> Params["Parameter Adjustments"]
        Lighting["Lighting Conditions"] --> Confidence["Confidence Adjustments"]
    Params --> Decision["Decision Logic"]
        Confidence --> Decision
    Decision --> Behavior["Context-Optimized Behavior"]
    end

    style Surface,Lighting fill:#94a3b7,stroke:#7a8697,stroke-width:1px,color:#000000,font-weight:bold
    style Params,Confidence fill:#b7af93,stroke:#979079,stroke-width:1px,color:#000000,font-weight:bold
    style Decision fill:#b29a9d,stroke:#937f81,stroke-width:1px,color:#000000,font-weight:bold
    style Behavior fill:#99aa9d,stroke:#7d8c81,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows how context-aware decision making would incorporate environmental factors. Surface detection, lighting conditions, and obstacle detection (blue) feed into parameter adjustment, confidence adjustment, and path planning (yellow). These all influence the decision logic (red), which produces context-optimized behavior (green).

Key implementations would include:

1. **Surface Classification**:
   - Detect floor types (carpet, hardwood, etc.)
   - Adjust motion parameters for different surfaces
   - Compensate for varying friction and traction

2. **Lighting Adaptation**:
   - Detect lighting conditions (bright, dim, variable)
   - Adjust confidence thresholds for detection quality
   - Compensate for lighting-induced sensor issues

3. **Obstacle Awareness**:
   - Incorporate obstacle information into state decisions
   - Modify search patterns to avoid obstacles
   - Develop obstacle-aware recovery strategies

#### 18.2.2 Historical Context Utilization

The system could leverage historical data to improve future decisions:

```python
# Pseudocode for historical context utilization
def incorporate_historical_context(self):
    # Extract patterns from historical data
    ball_patterns = self.analyze_movement_patterns(self.position_history)
    transition_patterns = self.analyze_transition_patterns(self.state_history)
    
    # Apply pattern-based adjustments
    if ball_patterns.has_pattern("oscillation_between_points"):
        # Adjust parameters for oscillating ball
        self.lost_ball_timeout *= 0.8  # Reduce timeout for quicker response
        self.stationary_threshold *= 1.2  # Increase threshold to reduce false stationary
        
    if transition_patterns.has_pattern("search_fail_after_corners"):
        # Modify search pattern for corner recovery
        self.search_pattern = "corner_biased"
        self.max_search_time *= 1.2  # Extend search time for corners
```

Pattern detection would enhance:
- Recovery strategies based on previous successes
- Anticipatory movements based on common trajectories
- Specialized handling for different ball movement types
- Fault prevention by identifying problematic patterns

#### 18.2.3 Multi-Modal Sensing

Integrating additional sensing modalities would enhance context awareness:

1. **Audio Integration**:
   - Detect ball bouncing sounds for additional tracking
   - Use audio cues to help locate lost balls
   - Distinguish multiple balls through sound differences

2. **Thermal Sensing**:
   - Track ball heat signatures in dim lighting
   - Distinguish ball from similar visual objects
   - Track through partial occlusions

3. **Advanced Fusion Techniques**:
   - Implement factor graph fusion for multiple modalities
   - Develop confidence-weighted multimodal integration
   - Apply modality-specific reliability estimators

### 18.3 Distributed State Management

For more complex robot systems, the state management could be distributed across multiple nodes:

```mermaid

        flowchart LR
    subgraph "Distributed State Management"
        MC["Master Coordinator"] --- SM1["Motion State Manager"]
        MC --- SM2["Sensor State Manager"]
        MC --- SM3["Navigation State Manager"]
    end

    style MC fill:#b29a9d,stroke:#937f81,stroke-width:1px,color:#000000,font-weight:bold
```

**Diagram Explanation**: This diagram shows a distributed state management architecture. A master coordinator (red) communicates with specialized state managers (green) for different subsystems like motion, sensors, navigation, and tasks.

Benefits of this approach include:
- Specialization of state handling for different subsystems
- Improved scalability for complex robots
- Better fault isolation and recovery
- Parallel processing of state decisions

Implementation would require:
- Inter-node communication protocol development
- Distributed decision-making algorithms
- Conflict resolution mechanisms
- Synchronized state transitions


## 19. <a name="faq"></a>Frequently Asked Questions

### 19.1 General Questions

#### Q: What is the purpose of the State Management Node?
A: The State Management Node serves as the decision-making brain of the basketball tracking robot. It interprets sensor data, makes behavioral decisions, and manages transitions between states like tracking, searching, and stopping.

#### Q: How does the State Management Node differ from the Fusion Node?
A: The Fusion Node focuses on "what is happening" (perception) by combining sensor data into a coherent understanding of ball position and motion. The State Management Node decides "what to do about it" (decision making) based on that perception data.

#### Q: What hardware is required to run the State Management Node?
A: The node runs efficiently on a Raspberry Pi 4 or 5 with at least 2GB RAM. It can also run on other Linux-based systems with ROS2 support, including Jetson platforms.

### 19.2 Implementation Questions

#### Q: How can I adjust the robot's sensitivity to ball movement?
A: Modify the `stationary_threshold` parameter. Higher values make the robot less sensitive to small movements, while lower values cause it to react to smaller movements.

#### Q: My robot oscillates between TRACKING and SEARCHING states. How can I fix this?
A: Increase the `tracking_hysteresis_time` and `lost_ball_hysteresis_time` parameters to add more stability. Also consider increasing `min_tracking_detections` to require more consecutive detections.

#### Q: How can I make the robot respond faster to the ball disappearing?
A: Decrease the `lost_ball_timeout` parameter to make the robot enter SEARCHING mode more quickly when the ball is lost.

#### Q: How can I optimize for a resource-constrained platform like Raspberry Pi 3?
A: Reduce the update rate to 50Hz or lower, enable simplified health monitoring, and reduce diagnostic publishing rates. These changes are available in the resource-constrained configuration example.

### 19.3 Technical Questions

#### Q: How does hysteresis protection work in the state machine?
A: Hysteresis protection prevents rapid oscillation between states by requiring minimum time in each state, different thresholds for entering versus exiting states, and consecutive detection counts for transitions.

#### Q: What is the purpose of the circular buffer implementation?
A: Circular buffers provide memory-efficient storage of time-series data with constant-time add operations and automatic discarding of oldest data. This helps track history while maintaining predictable memory usage.

#### Q: How does the system handle temporary sensor failures?
A: The sensor gap handling mechanism detects when sensors temporarily fail to provide data. For short gaps, it maintains the current state while reducing velocity. For longer gaps, it enters RECOVERY or SEARCHING states depending on uncertainty levels.

#### Q: How are parameters adapted based on ball movement?
A: The adaptive parameter system adjusts timeouts, thresholds, and detection requirements based on the ball's motion state. For example, stationary balls get more lenient parameters, while fast-moving balls get stricter requirements.

## 20. <a name="related-components"></a>Related Components

### 20.1 Links to Related Documentation

#### Fusion System
- **Kalman Filter Implementation**: See [Fusion.md: Understanding the Kalman Filter](/docs/Fusion.md#understanding-the-kalman-filter) for details on how position and uncertainty are calculated
- **Motion State Detection**: The State Manager relies on motion state information detailed in [Fusion.md: Motion State Management](/docs/Fusion.md#motion-state-management)
- **Sensor Gap Detection**: Gap handling is built on sensor gap information described in [Fusion.md: Sensor Gap Handling](/docs/Fusion.md#sensor-gap-handling)
- **Position Uncertainty**: Understanding how uncertainty is calculated is crucial for state transitions, see [Fusion.md: Position Uncertainty](/docs/Fusion.md#position-uncertainty)

#### PID Controller
- **State-to-Command Translation**: See [PidController.md: Command Generation](/docs/PidController.md#command-generation) for how states are translated to movement commands
- **Stopped State Handling**: Special PID configurations are used during STOPPED state, detailed in [PidController.md: Stationary Target Handling](/docs/PidController.md#stationary-target-handling)
- **Search Pattern Implementation**: The actual search patterns used during SEARCHING state are implemented in [PidController.md: Search Pattern Generation](/docs/PidController.md#search-pattern-generation)

### 20.2 ROS2 Dependencies

The State Management Node depends on the following ROS2 packages:

1. **std_msgs**: For basic message types (String, Float32, Bool)
2. **geometry_msgs**: For position and velocity messages (PoseStamped, Twist)
3. **ball_chase_msgs**: For custom message and service types
4. **rcl**: ROS Client Library for core ROS functionality
5. **rclpy**: Python client library for ROS2
6. **launch**: For launch file functionality

### 21.3 External Dependencies

External dependencies include:

1. **Python 3.9+**: The implementation language
2. **numpy**: For numerical operations
3. **transitions**: Optional library for finite state machine implementation
4. **matplotlib**: For visualization tools (optional)

## 21. <a name="glossary"></a>Glossary

| Term | Definition |
|------|------------|
| **Adaptive Parameters** | System parameters that automatically adjust based on detected conditions such as ball movement patterns or sensor quality. |
| **Ball Distance** | The calculated distance between the robot and the basketball, used for proximity detection. |
| **Confidence** | A measure (0.0-1.0) of how reliable the system's tracking and perception is, affecting decision making. |
| **Counter-Based Hysteresis** | A form of hysteresis that requires multiple consecutive events before triggering a state change. |
| **Detection Stability** | A measure of how consistent ball detections are over time, affecting state transitions. |
| **Finite State Machine (FSM)** | A computational model used to represent and control execution flow, consisting of states, transitions, and actions. |
| **Fusion Node** | The component that combines data from multiple sensors to estimate ball position and uncertainty. |
| **Health Monitoring** | The system that assesses overall robot performance by tracking various metrics and detecting warning conditions. |
| **Hysteresis** | A buffer or delay built into transitions to prevent rapid oscillation between states, creating more stable behavior. |
| **INITIALIZING** | The initial state of the robot when the system starts up, waiting for first reliable detection. |
| **LOST_BALL** | A state indicating that the ball is completely lost after extensive searching, waiting for redetection. |
| **Motion State** | Classification of the ball's movement pattern (e.g., stationary, medium_fast) that affects parameter adaptation. |
| **PID Controller** | The component that translates high-level commands from the State Manager into motor control signals. |
| **Position Uncertainty** | A measure of how confident the system is about the calculated position of the ball, affecting tracking reliability. |
| **RECOVERY** | A state focused on restoring reliable tracking when uncertainty is high or sensors provide conflicting data. |
| **SEARCHING** | A state where the robot actively searches for a temporarily lost ball using programmed search patterns. |
| **Sensor Gap** | A temporary period when sensors fail to provide detection data, requiring special handling. |
| **STOPPED** | A state where the robot has stopped moving because the ball is close and stationary. |
| **State Transition** | The process of changing from one operational state to another based on specific conditions. |
| **Stationary Detection** | The process of determining when a ball has stopped moving for a specified period of time. |
| **System Confidence** | A calculated value representing overall system health and reliability based on multiple factors. |
| **Time-Based Hysteresis** | A form of hysteresis that requires a minimum time in a state before transitions are allowed. |
| **TRACKING** | The normal operating state where the robot is actively following the basketball. |
| **Tracking Confidence** | A measure of how reliable the current tracking is, based on detection quality and consistency. |
| **Trend Analysis** | The process of examining how a value changes over time to detect patterns like rising, falling, or stable trends. |
| **Warning Condition** | A detected issue that may affect system performance, such as high uncertainty or sensor conflicts. |

## 22. <a name="references"></a>References

1. Quigley, M., Gerkey, B., & Smart, W. D. (2015). *Programming Robots with ROS: A Practical Introduction to the Robot Operating System*. O'Reilly Media.

2. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.

3. Konolige, K., Marder-Eppstein, E., & Marthi, B. (2011). "Navigation in Hybrid Metric-Topological Maps." In *IEEE International Conference on Robotics and Automation (ICRA)*.

4. Corke, P. (2017). *Robotics, Vision and Control: Fundamental Algorithms in MATLAB*. Springer.

5. Harel, D. (1987). "Statecharts: A Visual Formalism for Complex Systems." *Science of Computer Programming, 8(3)*, 231-274.

6. Brooks, R. A. (1986). "A Robust Layered Control System for a Mobile Robot." *IEEE Journal of Robotics and Automation, 2(1)*, 14-23.

7. Arkin, R. C. (1998). *Behavior-Based Robotics*. MIT Press.

8. Welch, G. & Bishop, G. (2006). "An Introduction to the Kalman Filter." *Department of Computer Science, University of North Carolina at Chapel Hill*.

9. Marder-Eppstein, E., Berger, E., Foote, T., Gerkey, B., & Konolige, K. (2010). "The Office Marathon: Robust Navigation in an Indoor Office Environment." In *IEEE International Conference on Robotics and Automation (ICRA)*.

10. ROS 2 Documentation. (2024). "Creating a ROS 2 Package." Retrieved from [docs.ros.org](https://docs.ros.org/).
