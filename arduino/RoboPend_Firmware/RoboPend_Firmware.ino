/*
 * ============================================================
 *  RoboPend_Firmware.ino  — v4.1
 * ============================================================
 *
 *  PHYSICAL WIRING:
 *    D2  ← Pendulum encoder A  (hardware interrupt INT0)
 *    D3  ← Pendulum encoder B
 *    D7  ← Cart encoder A      (Pin Change Interrupt PCINT23)
 *    D8  ← Cart encoder B
 *    D11 → Stepper PULSE (STEP)
 *    D10 → Stepper DIR
 *    (No EN pin — driver is always enabled)
 *
 *  ENCODER CONVENTION:
 *    Pendulum: 0 counts = hanging, +1200 = upright, 2400 = full rev
 *    Cart:     0 counts = zeroed at calibration, ±14500 ≈ rail ends
 *
 *  SERIAL PROTOCOL  (115200 baud):
 *    Python → Arduino:
 *      "V<float>\n"   — set motor velocity in steps/sec
 *      "H\n"          — stop motor (velocity = 0)
 *      "Z\n"          — zero BOTH encoder counters in place
 *
 *    Arduino → Python:
 *      "P <pend> <cart>\n"   — sent at 100 Hz
 *      "LIMIT\n"             — cart exceeded safe boundary
 *      "READY\n"             — sent once on boot
 * ============================================================
 */

#include <AccelStepper.h>

// ── Pin Definitions ──────────────────────────────────────────
const int PEND_ENC_A = 2;   // Hardware interrupt INT0
const int PEND_ENC_B = 3;
const int CART_ENC_A = 7;   // Pin Change Interrupt (PCINT23)
const int CART_ENC_B = 8;

const int STEP_PIN = 11;  // PULSE
const int DIR_PIN  = 10;  // DIR
// No EN pin — TB6600 driver is always enabled

// ── Configuration ────────────────────────────────────────────
const long  SAFE_LIMIT_TICKS = 14000;  // half of 29000 rail, minus margin
const float MAX_SPEED_STEPS  = 4000.0; // steps/sec
const int   REPORT_PERIOD_MS = 10;     // 100 Hz telemetry

// ── State ────────────────────────────────────────────────────
volatile long pend_ticks = 0;
volatile long cart_ticks = 0;

float         target_velocity = 0.0;
unsigned long last_report_ms  = 0;

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// ── Encoder ISRs ─────────────────────────────────────────────
void pendulumISR() {
  if (digitalRead(PEND_ENC_A) == digitalRead(PEND_ENC_B))
    pend_ticks++;
  else
    pend_ticks--;
}

// Cart on pin 7 (PCINT23, Port D bit 7) — not a hardware interrupt pin
ISR(PCINT2_vect) {
  if (digitalRead(CART_ENC_A) == digitalRead(CART_ENC_B))
    cart_ticks++;
  else
    cart_ticks--;
}

// ── Atomic reads ─────────────────────────────────────────────
long readPend() { noInterrupts(); long v = pend_ticks; interrupts(); return v; }
long readCart() { noInterrupts(); long v = cart_ticks; interrupts(); return v; }

// ── Setup ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // Encoder pins
  pinMode(PEND_ENC_A, INPUT_PULLUP);
  pinMode(PEND_ENC_B, INPUT_PULLUP);
  pinMode(CART_ENC_A, INPUT_PULLUP);
  pinMode(CART_ENC_B, INPUT_PULLUP);

  // Pendulum encoder — hardware interrupt on D2
  attachInterrupt(digitalPinToInterrupt(PEND_ENC_A), pendulumISR, CHANGE);

  // Cart encoder on D7 — Pin Change Interrupt
  PCICR  |= (1 << PCIE2);    // enable PCINT for Port D
  PCMSK2 |= (1 << PCINT23);  // unmask D7 (PD7 = PCINT23)

  // Stepper — no EN pin, driver always on
  stepper.setMinPulseWidth(20);
  stepper.setMaxSpeed(MAX_SPEED_STEPS);

  Serial.println(F("READY"));
}

// ── Main Loop ─────────────────────────────────────────────────
void loop() {

  // 1. Parse incoming serial command
  if (Serial.available() > 0) {
    char cmd = Serial.peek();

    if (cmd == 'V') {
      Serial.read();
      target_velocity = Serial.parseFloat();
    }
    else if (cmd == 'H') {
      Serial.read();
      target_velocity = 0.0;
    }
    else if (cmd == 'Z') {
      Serial.read();
      noInterrupts();
      pend_ticks = 0;
      cart_ticks = 0;
      interrupts();
      target_velocity = 0.0;
    }
    else {
      Serial.read();  // discard unknown bytes
    }
  }

  // 2. Safety limit check — just stop, no disable (no EN pin)
  long c = readCart();
  if (abs(c) > SAFE_LIMIT_TICKS) {
    target_velocity = 0.0;
    stepper.setSpeed(0);
    stepper.runSpeed();
    Serial.println(F("LIMIT"));
    return;
  }

  // 3. Drive stepper
  stepper.setSpeed(target_velocity);
  stepper.runSpeed();

  // 4. Telemetry at 100 Hz — "P <pend> <cart>"
  unsigned long now = millis();
  if (now - last_report_ms >= REPORT_PERIOD_MS) {
    last_report_ms = now;
    Serial.print(F("P "));
    Serial.print(readPend());
    Serial.print(F(" "));
    Serial.println(readCart());
  }
}
