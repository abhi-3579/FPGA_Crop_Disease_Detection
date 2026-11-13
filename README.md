FPGA-Based Crop Disease Detection Using CNN
Project Overview
This project demonstrates CNN-based crop leaf disease classification using Verilog HDL and a Digilent Nexys A7 FPGA development board. A Python host application supplies image data, communicates with the FPGA through UART, and receives a classification result that is displayed on the board's seven-segment display.

The exact preprocessing, data format, CNN architecture, and UART protocol must match the implementation in the source files.
Objectives
- Implement CNN inference logic in Verilog HDL.
- Integrate the CNN core with UART communication.
- Transfer input data between a host computer and the FPGA.
- Display the predicted class index on the Nexys A7 seven-segment display.
- Verify hardware results against a software reference.
Hardware and Software Requirements
Hardware
- Digilent Nexys A7 FPGA development board
- Computer/laptop
- USB cable for programming and serial communication
Software
- Xilinx Vivado Design Suite
- Python 3
- Python packages required by the supplied scripts
- Serial communication library or terminal, if required by the implementation
Repository Structure
FPGA_Crop_Disease_Detection/
├── cnn_core.v          # CNN processing core
├── uart_seg.v          # UART communication and seven-segment display logic
├── top.v               # Top-level FPGA module
├── tb_cnn_core.v       # CNN core testbench
├── nexys_a7.xdc        # Nexys A7 pin constraints
├── gen_hw.py           # Hardware-data/parameter generation script
├── verify_twin.py      # Verification/reference comparison script
└── README.md           # Project documentation
Working Principle
1. Image input: A crop leaf image is supplied through the Python host-side application.
2. Preprocessing: The image is converted into the data format expected by the hardware.
3. UART transfer: The host application sends the required data to the FPGA using the implemented UART protocol.
4. CNN processing: The Verilog CNN core processes the input and generates a classification result.
5. Result display: The result is sent to the display logic and shown as a class index on the Nexys A7 seven-segment display.
6. Verification: Hardware results can be compared with a software reference using the verification script.
RTL Module Description
cnn_core.v
Contains the CNN processing logic implemented in Verilog HDL. It performs the hardware computations used by the classifier and produces the predicted class result.
uart_seg.v
Contains UART communication and seven-segment display-related logic used to transfer and present the classification output.
top.v
Top-level module that connects the design blocks, clock/reset signals, communication signals, and display outputs.
tb_cnn_core.v
Simulation testbench for applying test inputs to the CNN core and observing its outputs.
nexys_a7.xdc
Contains FPGA pin assignments and constraints for the Nexys A7 board. Confirm that the selected board variant matches the constraints.
Python Script Description
gen_hw.py
Supports generation of hardware-related data or parameters used by the FPGA implementation. Check the script for its exact outputs and required arguments.
verify_twin.py
Supports verification or comparison between a software reference and the hardware implementation. The exact checks depend on the script configuration.
Implementation Flow
Crop Leaf Image
       |
       v
Python Host Application
       |
       v
Image Preprocessing
       |
       v
UART Data Transfer
       |
       v
Nexys A7 FPGA
       |
       v
Verilog CNN Core
       |
       v
Classification Result
       |
       v
Seven-Segment Display
Disease Classification Demonstration
The demonstration images have been associated with these disease labels:
Crop	Disease label
Pepper Bell	Bacterial Spot
Strawberry	Leaf Scorch
Tomato	Early Blight


The seven-segment display shows a model-specific class index. The mapping between class indices and disease labels must be checked against the class order used during model training and in the implementation.

Important: A model prediction is not laboratory confirmation of a disease or pathogen. Confirm the disease labels against the actual dataset and trained model before treating them as verified results.
How to Run
1. Clone or download this repository.
2. Open Vivado and create a project for the correct Nexys A7 board variant.
3. Add cnn_core.v, uart_seg.v, and top.v as design sources.
4. Add nexys_a7.xdc as the constraints file.
5. Set top as the top-level module.
6. Add tb_cnn_core.v as a simulation source and run the testbench.
7. Run synthesis and implementation, review any warnings, and generate the bitstream.
8. Program the FPGA board.
9. Install the Python dependencies required by the scripts and review their command-line arguments/configuration.
10. Run the host-side flow, provide a test image, and observe the classification output.
Before running the design, confirm that the clock frequency, UART baud rate, reset polarity, pin assignments, data format, and Python configuration match the RTL.
Verification and Results
The CNN core can be simulated using tb_cnn_core.v. Use verify_twin.py for software/hardware comparison if the script is configured for that purpose.

For a complete project report, include actual simulation waveforms, FPGA board photographs, input leaf images, displayed class indices, and measured results. Do not claim accuracy, latency, throughput, or power consumption unless these values have been measured and verified.
Applications
- Crop leaf disease classification demonstrations
- FPGA-based CNN inference
- Hardware acceleration for embedded image-classification systems
- Educational work in Verilog HDL, digital design, and FPGA development
Future Enhancements
- Add more crop types and disease classes.
- Improve preprocessing and classification performance.
- Measure FPGA resource utilization, operating frequency, latency, and power.
- Develop a user-friendly host-side interface.
- Explore CNN architectures optimized for FPGA implementation.
Author
Kunda Abhilasha
Acknowledgement
This project demonstrates CNN-based crop disease classification using Verilog HDL, Python, and the Digilent Nexys A7 FPGA development board.
