# FPGA-Based Crop Disease Detection Using CNN

## 1. Project Overview

The FPGA-Based Crop Disease Detection System is designed to identify crop leaf diseases using Convolutional Neural Networks (CNN) implemented with Verilog HDL. The system uses the Digilent Nexys A7 FPGA development board to perform hardware-based classification.

A crop leaf image is provided through a Python-based host application. The input data is processed and transferred to the FPGA through UART communication. The CNN core performs classification, and the predicted class index is displayed on the seven-segment display of the FPGA board.

The main objective is to demonstrate the implementation of CNN-based image classification using FPGA hardware acceleration.

## 2. Objectives

- To implement CNN processing using Verilog HDL.
- To perform crop leaf disease classification.
- To integrate UART communication between the computer and FPGA.
- To display classification results using a seven-segment display.
- To verify the hardware output using Python-based verification.

## 3. Hardware Requirements

- Digilent Nexys A7 FPGA Development Board
- Computer or Laptop
- USB Cable

## 4. Software Requirements

- Xilinx Vivado Design Suite
- Verilog HDL
- Python 3
- Required Python Libraries

## 5. Project File Structure

The project consists of the following files:

**1. cnn_core.v**

Implements the CNN processing core using Verilog HDL. It performs the hardware computations required for image classification.

**2. uart_seg.v**

Implements UART communication and seven-segment display-related logic for transferring and displaying classification results.

**3. top.v**

The top-level module that integrates the CNN core, UART communication, and display-related modules.

**4. tb_cnn_core.v**

The testbench used to verify the functionality of the CNN core through simulation.

**5. nexys_a7.xdc**

Contains the FPGA pin assignments and constraints required for the Nexys A7 development board.

**6. gen_hw.py**

Python script used for generating hardware-related data or parameters required by the implementation.

**7. verify_twin.py**

Python script used for verification and comparison between reference and hardware results.

## 6. System Architecture

The system follows the sequence below:

1. Crop leaf image acquisition.
2. Image preprocessing using Python.
3. Data transfer through UART communication.
4. CNN-based processing inside the FPGA.
5. Classification result generation.
6. Display of the predicted class index on the seven-segment display.

## 7. Implementation Procedure

1. Develop the CNN processing logic using Verilog HDL.
2. Integrate the required modules using the top-level module.
3. Verify the CNN core using the Verilog testbench.
4. Add the Nexys A7 constraints file in Vivado.
5. Perform synthesis and implementation.
6. Generate the FPGA bitstream.
7. Program the Nexys A7 development board.
8. Execute the Python-based host application.
9. Provide crop leaf images and observe the classification output.

## 8. Disease Classification

The demonstration includes the following crop diseases:

| S.No. | Crop Plant | Disease |
|---|---|---|
| 1 | Pepper Bell | Bacterial Spot |
| 2 | Strawberry | Leaf Scorch |
| 3 | Tomato | Early Blight |

The FPGA displays the corresponding class index based on the trained CNN model. The class index mapping depends on the model's training configuration.

## 9. Results

The proposed system demonstrates the integration of CNN processing, UART communication, and FPGA-based classification. The Nexys A7 board displays the predicted class index through its seven-segment display.

The experimental demonstration shows how FPGA hardware can be used for crop leaf disease classification.

## 10. Applications

- Smart Agriculture
- Crop Disease Identification
- FPGA-Based Image Classification
- Embedded Artificial Intelligence
- Hardware Acceleration

## 11. Future Scope

- Support for additional crop diseases.
- Improve classification performance.
- Optimize FPGA resource utilization.
- Reduce processing latency and power consumption.
- Integrate the system with real-time agricultural monitoring applications.

## 12. Author

**Kunda Abhilasha**

**Project Domain:** VLSI Design and FPGA-Based Image Processing

**Hardware Platform:** Digilent Nexys A7 FPGA

**Programming Languages:** Verilog HDL and Python
