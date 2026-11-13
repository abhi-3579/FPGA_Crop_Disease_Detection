
`timescale 1ns/1ps
`include "cnn_params.vh"

module tb_cnn_image;

  reg clk = 0;
  reg rst = 1;
  always #5 clk = ~clk;  // 100 MHz

  reg start = 0;
  reg finish = 0;
  reg ld_we = 0;

  reg [`AW-1:0] ld_addr = 0;
  reg [`AW-1:0] ro_addr = 0;
  reg [7:0] ld_data = 0;

  wire [7:0] ro_data;
  wire [7:0] class_idx;
  wire done;

  // CNN core
  cnn_core dut (
    .clk(clk),
    .rst(rst),
    .start(start),
    .finish(finish),
    .ld_we(ld_we),
    .ld_addr(ld_addr),
    .ld_data(ld_data),
    .ro_addr(ro_addr),
    .ro_data(ro_data),
    .done(done),
    .class_idx(class_idx)
  );

  // Image memory: 64x64x3 = 12288 bytes
  reg [7:0] image [0:`IMG_BYTES-1];

  integer i;
  integer predicted_class;
  integer max_logit;

  reg signed [7:0] current_logit;

  initial begin

    $display("");
    $display("==========================================");
    $display("       TINYML CROP DISEASE CNN TEST");
    $display("==========================================");

    // ======================================
    // STEP 1: LOAD IMAGE HEX FILE
    // ======================================

    $display("Loading input image HEX file...");

    // CHANGE THIS FILENAME TO YOUR IMAGE HEX FILE
    $readmemh("/home/hussain/Music/tomato_leaf.hex", image);

    $display("Image HEX loaded.");
    $display("Image size: %0d bytes", `IMG_BYTES);

    // ======================================
    // STEP 2: RESET
    // ======================================

    #100;
    rst = 0;

    repeat(5) @(posedge clk);

    // ======================================
    // STEP 3: LOAD IMAGE INTO CNN
    // ======================================

    $display("Loading image into CNN memory...");

    for (i = 0; i < `IMG_BYTES; i = i + 1) begin
      @(negedge clk);
      ld_we = 1'b1;
      ld_addr = i;
      ld_data = image[i];
    end

    @(negedge clk);
    ld_we = 0;

    $display("Image memory loading completed.");

    // ======================================
    // STEP 4: START CNN
    // ======================================

    $display("Starting CNN inference...");

    @(negedge clk);
    start = 1;

    @(negedge clk);
    start = 0;

    // ======================================
    // STEP 5: WAIT FOR CNN COMPLETION
    // ======================================

    $display("CNN processing...");

    wait(done == 1'b1);

    $display("");
    $display("CNN inference completed!");
    $display("Hardware predicted class = %0d", class_idx);

    // ======================================
    // STEP 6: READ ALL 38 LOGITS
    // ======================================

    max_logit = -128;
    predicted_class = 0;

    $display("");
    $display("==========================================");
    $display("             CNN OUTPUT LOGITS");
    $display("==========================================");

    for (i = 0; i < `NUM_CLASSES; i = i + 1) begin

      @(negedge clk);
      ro_addr = i;

      // Synchronous memory read + registered read select
      repeat(3) @(posedge clk);
      #1;

      current_logit = $signed(ro_data);

      $display("Class %0d : Logit = %0d (HEX = %h)",
               i, current_logit, ro_data);

      if (current_logit > max_logit) begin
        max_logit = current_logit;
        predicted_class = i;
      end

    end

    // ======================================
    // STEP 7: FINAL RESULT
    // ======================================

    $display("");
    $display("==========================================");
    $display("             FINAL RESULT");
    $display("==========================================");

    $display("Predicted Class Index : %0d", class_idx);
    $display("Maximum Logit         : %0d", max_logit);
    $display("Argmax Class          : %0d", predicted_class);

    if (class_idx == predicted_class)
      $display("CNN VERIFICATION : PASS");
    else
      $display("CNN VERIFICATION : FAIL");

    $display("==========================================");

    // Release CNN core back to idle
    @(negedge clk);
    finish = 1;

    @(negedge clk);
    finish = 0;

    $finish;

  end

  // Simulation timeout
  initial begin
    #500000000;
    $display("ERROR: CNN SIMULATION TIMEOUT");
    $finish;
  end

endmodule
