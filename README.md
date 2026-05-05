# RoboPend

A high-performance Reinforcement Learning controller for a NEMA 17 Pendulum system.

## 📁 Repository Structure

### 🧠 The Brain
*   **`pendulum_final.zip`**: The trained SAC model (+2200 reward). 100% success rate in simulation.

### 🔌 Hardware Deployment
*   **`arduino/`**: Contains the firmware for the motor driver and encoders.
*   **`verify_hardware.py`**: **Run this first.** Safety script to test physical encoder readings and motor response.
*   **`run_inference.py`**: The main script to run the RL model on the physical hardware.
*   **`Hardware_Env.py`**: Low-level communication bridge between Python and Arduino.

### 🔬 Simulation & Training
*   **`demo_best.py`**: Run this to see the current model perform in the PyBullet simulation.
*   **`RL_train.py`**: The master training script used to generate the models.
*   **`main.py`**: Manual GUI test to verify PyBullet physics and URDF joints.
*   **`Pend_assem_stl/`**: 3D models and URDF definition.

## 🚀 Quick Start
1. Flash Arduino with `arduino/RoboPend_Firmware/`.
2. Connect hardware and run `python verify_hardware.py`.
3. If encoders are correct, run `python run_inference.py`.
