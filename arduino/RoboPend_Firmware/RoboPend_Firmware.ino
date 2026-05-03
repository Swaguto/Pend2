/*
 * RoboPend_Firmware.ino
 * 
 * Hardware Interface for Pendulum Sim2Real.
 * - Reads 2x LPD3806 Incremental Encoders (600 PPR -> 2400 counts/rev)
 * - Controls NEMA 17 Stepper via TB6600 (STEP/DIR/EN)
 * - Communicates with Python via Serial (115200 baud)
 */

#include <AccelStepper.h>

// --- PIN DEFINITIONS (Adjust these to your physical wiring) ---
const int CART_ENC_A = 2;   // Interrupt Pin
const int CART_ENC_B = 4;
const int PEND_ENC_A = 3;   // Interrupt Pin
const int PEND_ENC_B = 5;

const int STEP_PIN = 9;
const int DIR_PIN  = 8;
const int EN_PIN   = 10;

// --- CONFIGURATION ---
const long SAFE_LIMIT_TICKS = 15000; // Adjust based on your rail length
const int  SERIAL_FREQ_MS   = 10;    // Send data to Python at 100Hz

// --- STATE ---
volatile long cart_pos_ticks = 0;
volatile long pend_pos_ticks = 0;
unsigned long last_serial_ms = 0;

// Stepper Interface
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// --- INTERRUPT HANDLERS ---
void handleCartEnc() {
  if (digitalRead(CART_ENC_A) == digitalRead(CART_ENC_B)) cart_pos_ticks++;
  else cart_pos_ticks--;
}

void handlePendEnc() {
  if (digitalRead(PEND_ENC_A) == digitalRead(PEND_ENC_B)) pend_pos_ticks++;
  else pend_pos_ticks--;
}

void setup() {
  Serial.begin(115200);

  // Encoder Pins
  pinMode(CART_ENC_A, INPUT_PULLUP);
  pinMode(CART_ENC_B, INPUT_PULLUP);
  pinMode(PEND_ENC_A, INPUT_PULLUP);
  pinMode(PEND_ENC_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(CART_ENC_A), handleCartEnc, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PEND_ENC_A), handlePendEnc, CHANGE);

  // Stepper Setup
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW); // Enable motor (usually active LOW)
  
  stepper.setMaxSpeed(4000);     // Adjust based on your TB6600 microstepping
  stepper.setAcceleration(20000); // We will update velocity dynamically
}

void loop() {
  // 1. SAFETY CHECK
  if (abs(cart_pos_ticks) > SAFE_LIMIT_TICKS) {
    digitalWrite(EN_PIN, HIGH); // DISABLE MOTOR
    stepper.stop();
  }

  // 2. RECEIVE COMMANDS FROM PYTHON
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'V') { // Velocity Command
      float target_vel = Serial.parseFloat();
      stepper.setSpeed(target_vel);
    } 
    else if (cmd == 'H') { // Home/Reset Command
      cart_pos_ticks = 0;
      pend_pos_ticks = 0;
      digitalWrite(EN_PIN, LOW); // Re-enable
    }
  }

  // 3. STEP THE MOTOR
  stepper.runSpeed();

  // 4. TRANSMIT DATA TO PYTHON (100Hz)
  if (millis() - last_serial_ms >= SERIAL_FREQ_MS) {
    last_serial_ms = millis();
    
    // Output format: P <cart> <pend>
    Serial.print("P ");
    Serial.print(cart_pos_ticks);
    Serial.print(" ");
    Serial.println(pend_pos_ticks);
  }
}
