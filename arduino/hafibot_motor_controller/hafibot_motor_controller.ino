/*
 * HAFIBOT Motor Controller v4.0 - Arduino Mega 2560 Pro Mini
 *
 * Protocol v2 from Jetson (115200 baud):
 *   <mode,distance_cm,center_x,detected>\n
 *   F = follow, S = soft stop, E = latched fall ESTOP, R = reset ESTOP
 *
 * Telemetry to Jetson:
 *   S,STATE,usLeft,usCenter,usRight,fallEstopLatched
 *
 * Safety ownership stays on this controller: malformed/stale commands stop,
 * fall ESTOP requires an explicit reset, and obstacle avoidance does not rely
 * on ROS latency.  Reverse following is intentionally disabled because this
 * chassis has no rear obstacle sensors.
 */

#include <NewPing.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// 4WD motor pins: L298N #1 drives the left side; #2 drives the right side.
// ---------------------------------------------------------------------------
#define ENA_L   6
#define IN1_LF  2
#define IN2_LF  3
#define IN3_LR  4
#define IN4_LR  5
#define ENB_L   7

#define ENA_R   8
#define IN1_RF  22
#define IN2_RF  23
#define IN3_RR  24
#define IN4_RR  25
#define ENB_R   9

// Front-facing ultrasonic sensors.
#define TRIG_LEFT   A0
#define ECHO_LEFT   A1
#define TRIG_CENTER A2
#define ECHO_CENTER A3
#define TRIG_RIGHT  A4
#define ECHO_RIGHT  A5

const int MAX_SONAR_DISTANCE = 200;
NewPing sonarLeft(TRIG_LEFT, ECHO_LEFT, MAX_SONAR_DISTANCE);
NewPing sonarCenter(TRIG_CENTER, ECHO_CENTER, MAX_SONAR_DISTANCE);
NewPing sonarRight(TRIG_RIGHT, ECHO_RIGHT, MAX_SONAR_DISTANCE);

// ---------------------------------------------------------------------------
// Tunable motion and safety parameters.
// ---------------------------------------------------------------------------
// Tunable motion parameters calibrated for 3S LiPo (~11.1V - 12.6V) + dual L298N (star topology) + 1.7 kg 4WD
const int SPEED_FORWARD_MIN = 180;  // High starting threshold to guarantee rotation under L298N Vdrop
const int SPEED_FORWARD_MAX = 255;  // Full 100% duty cycle available for following
const int SPEED_TURN_MIN = 140;     // Skid-steer rotational friction breakaway on 3S
const int SPEED_TURN_MAX = 220;     // Responsive skid-steer turning limit
const int SPEED_AVOID = 200;        // Active pivot avoidance speed
const int PWM_RAMP_STEP = 25;       // Fast ramp-up (reaches full power in ~100ms)
const int KICKSTART_FEEDFORWARD_PWM = 255; // 100% full-power initial kick to overcome stiction
const unsigned long KICKSTART_FEEDFORWARD_DURATION_MS = 150; // 150 ms stiction breakaway duration

// Human following distance parameters:
// Robot stays in range at 130-175 cm, moves forward when user is > 175 cm away,
// stops approaching at 130 cm, and stops immediately if closer than 80 cm.
const int DIST_TOO_CLOSE = 80;
const int DIST_FORWARD_STOP = 130;
const int DIST_FORWARD_START = 175;
const int DIST_ESTIMATOR_MAX = 450;

// Heading Alignment PID Controller Constants:
const float TURN_KP = 150.0f;       // Proportional gain
const float TURN_KI = 5.0f;         // Integral gain (steady state offset)
const float TURN_KD = 30.0f;        // Derivative gain (anti-overshoot damping brake)
const float TURN_INTEGRAL_LIMIT = 0.20f;
const float CX_DEADBAND = 0.035f;   // Inside 0.465 - 0.535, keep wheels straight
const float CX_PIVOT_ERROR = 0.25f; // Large error: slow pivot in place

const int OA_TRIGGER_CM = 25;
const int OA_CLEAR_CM = 35;
const unsigned long OA_STOP_DWELL_MS = 200;
const unsigned long OA_CLEAR_CONFIRM_MS = 300;

const unsigned long SERIAL_TIMEOUT_MS = 1500;
const unsigned long TELEMETRY_INTERVAL_MS = 200;
const unsigned long US_READ_INTERVAL_MS = 100;
const unsigned long CONTROL_INTERVAL_MS = 50;
const unsigned long MOTOR_RAMP_INTERVAL_MS = 25;

enum RobotState {
  STATE_IDLE,
  STATE_DISABLED,
  STATE_FORWARD,
  STATE_ARC_LEFT,
  STATE_ARC_RIGHT,
  STATE_TURN_LEFT,
  STATE_TURN_RIGHT,
  STATE_IN_RANGE,
  STATE_TOO_CLOSE,
  STATE_AVOID_LEFT,
  STATE_AVOID_RIGHT,
  STATE_FALL_ESTOP,
  STATE_LINK_ESTOP
};

RobotState currentState = STATE_IDLE;

// Last valid Jetson command.
char commandMode = 'S';
int targetDistance = 0;
float targetCenterX = 0.5f;
bool targetDetected = false;
bool fallEstopLatched = false;
unsigned long lastDataTime = 0;

// Sensor values.  MAX+1 means no echo inside the configured sonar range.
int usLeft = MAX_SONAR_DISTANCE + 1;
int usCenter = MAX_SONAR_DISTANCE + 1;
int usRight = MAX_SONAR_DISTANCE + 1;

// Smoothed drive commands, signed: positive=forward, negative=reverse.
int targetLeftPower = 0;
int targetRightPower = 0;
int currentLeftPower = 0;
int currentRightPower = 0;

bool forwardActive = false;
bool prevForwardActive = false;
bool turnActive = false;
unsigned long forwardStartedAt = 0;

// Heading alignment PID controller state
float turnIntegral = 0.0f;
float lastHorizontalError = 0.0f;
unsigned long lastPidTime = 0;

// Obstacle avoidance is latched long enough to stop before beginning a turn.
bool obstacleActive = false;
int obstacleDirection = 0;  // -1=left, +1=right
unsigned long obstacleStartedAt = 0;
unsigned long obstacleClearSince = 0;

// Timers.
unsigned long lastTelemetryTime = 0;
unsigned long lastUSReadTime = 0;
unsigned long lastControlTime = 0;
unsigned long lastMotorRampTime = 0;

// Strict framed serial receiver.
const byte MAX_CHARS = 64;
char receivedChars[MAX_CHARS];
bool receiveInProgress = false;
bool receiveOverflow = false;
byte receiveIndex = 0;

// ---------------------------------------------------------------------------
// Motor primitives.
// ---------------------------------------------------------------------------
int clampPower(int value) {
  if (value > 255) return 255;
  if (value < -255) return -255;
  return value;
}

void writeLeftMotors(int signedPower) {
  int power = abs(clampPower(signedPower));
  analogWrite(ENA_L, power);
  analogWrite(ENB_L, power);
  if (signedPower > 0) {
    digitalWrite(IN1_LF, HIGH); digitalWrite(IN2_LF, LOW);
    digitalWrite(IN3_LR, HIGH); digitalWrite(IN4_LR, LOW);
  } else if (signedPower < 0) {
    digitalWrite(IN1_LF, LOW); digitalWrite(IN2_LF, HIGH);
    digitalWrite(IN3_LR, LOW); digitalWrite(IN4_LR, HIGH);
  } else {
    digitalWrite(IN1_LF, LOW); digitalWrite(IN2_LF, LOW);
    digitalWrite(IN3_LR, LOW); digitalWrite(IN4_LR, LOW);
  }
}

void writeRightMotors(int signedPower) {
  int power = abs(clampPower(signedPower));
  analogWrite(ENA_R, power);
  analogWrite(ENB_R, power);
  if (signedPower > 0) {
    digitalWrite(IN1_RF, HIGH); digitalWrite(IN2_RF, LOW);
    digitalWrite(IN3_RR, HIGH); digitalWrite(IN4_RR, LOW);
  } else if (signedPower < 0) {
    digitalWrite(IN1_RF, LOW); digitalWrite(IN2_RF, HIGH);
    digitalWrite(IN3_RR, LOW); digitalWrite(IN4_RR, HIGH);
  } else {
    digitalWrite(IN1_RF, LOW); digitalWrite(IN2_RF, LOW);
    digitalWrite(IN3_RR, LOW); digitalWrite(IN4_RR, LOW);
  }
}

void setDriveTarget(int leftPower, int rightPower) {
  targetLeftPower = clampPower(leftPower);
  targetRightPower = clampPower(rightPower);
}

void immediateStop() {
  targetLeftPower = 0;
  targetRightPower = 0;
  currentLeftPower = 0;
  currentRightPower = 0;
  writeLeftMotors(0);
  writeRightMotors(0);
}

int rampPower(int currentPower, int requestedPower) {
  if (requestedPower == 0) return 0; // Immediate cutoff when stopping

  // When starting from complete stop, jump directly past deadband floor to eliminate stall buzzing
  if (currentPower == 0 && requestedPower > 0) {
    return max(SPEED_FORWARD_MIN, requestedPower);
  }
  if (currentPower == 0 && requestedPower < 0) {
    return min(-SPEED_FORWARD_MIN, requestedPower);
  }

  // Always pass through zero before reversing a motor side.
  if (currentPower > 0 && requestedPower < 0) {
    return max(0, currentPower - PWM_RAMP_STEP);
  }
  if (currentPower < 0 && requestedPower > 0) {
    return min(0, currentPower + PWM_RAMP_STEP);
  }
  if (currentPower < requestedPower) {
    return min(requestedPower, currentPower + PWM_RAMP_STEP);
  }
  if (currentPower > requestedPower) {
    return max(requestedPower, currentPower - PWM_RAMP_STEP);
  }
  return currentPower;
}

void serviceMotorRamp() {
  currentLeftPower = rampPower(currentLeftPower, targetLeftPower);
  currentRightPower = rampPower(currentRightPower, targetRightPower);
  writeLeftMotors(currentLeftPower);
  writeRightMotors(currentRightPower);
}

void resetMotionIntent() {
  forwardActive = false;
  prevForwardActive = false;
  turnActive = false;
  forwardStartedAt = 0;
  obstacleActive = false;
  obstacleClearSince = 0;
  turnIntegral = 0.0f;
  lastHorizontalError = 0.0f;
}

// ---------------------------------------------------------------------------
// Ultrasonic sensing.
// ---------------------------------------------------------------------------
int readSonar(NewPing &sonar) {
  int cm = sonar.ping_cm();
  return cm == 0 ? MAX_SONAR_DISTANCE + 1 : cm;
}

void updateUltrasonic() {
  usLeft = readSonar(sonarLeft);
  delay(15);
  usCenter = readSonar(sonarCenter);
  delay(15);
  usRight = readSonar(sonarRight);
}

// ---------------------------------------------------------------------------
// Strict protocol-v2 parsing.
// ---------------------------------------------------------------------------
bool parseLongStrict(char *token, long minimum, long maximum, long &output) {
  if (token == NULL || *token == '\0') return false;
  char *endPointer = NULL;
  long value = strtol(token, &endPointer, 10);
  if (endPointer == token || *endPointer != '\0') return false;
  if (value < minimum || value > maximum) return false;
  output = value;
  return true;
}

bool parseFloatStrict(char *token, float minimum, float maximum, float &output) {
  if (token == NULL || *token == '\0') return false;
  char *endPointer = NULL;
  double value = strtod(token, &endPointer);
  if (endPointer == token || *endPointer != '\0' || !isfinite(value)) return false;
  if (value < minimum || value > maximum) return false;
  output = (float)value;
  return true;
}

bool parseData() {
  char *savePointer = NULL;
  char *modeToken = strtok_r(receivedChars, ",", &savePointer);
  char *distanceToken = strtok_r(NULL, ",", &savePointer);
  char *centerToken = strtok_r(NULL, ",", &savePointer);
  char *detectedToken = strtok_r(NULL, ",", &savePointer);
  char *extraToken = strtok_r(NULL, ",", &savePointer);

  if (modeToken == NULL || strlen(modeToken) != 1 || extraToken != NULL) {
    return false;
  }

  char mode = modeToken[0];
  if (mode != 'F' && mode != 'S' && mode != 'E' && mode != 'R') {
    return false;
  }

  long distanceValue = 0;
  long detectedValue = 0;
  float centerValue = 0.5f;
  if (!parseLongStrict(distanceToken, 0, 500, distanceValue)) return false;
  if (!parseFloatStrict(centerToken, 0.0f, 1.0f, centerValue)) return false;
  if (!parseLongStrict(detectedToken, 0, 1, detectedValue)) return false;

  // Non-follow commands must not carry a target.
  if (mode != 'F' && (distanceValue != 0 || detectedValue != 0)) return false;
  if (mode == 'F' && detectedValue == 1 && distanceValue == 0) return false;
  if (mode == 'F' && detectedValue == 0 && distanceValue != 0) return false;

  lastDataTime = millis();
  if (mode == 'E') {
    fallEstopLatched = true;
    commandMode = 'E';
    targetDistance = 0;
    targetDetected = false;
    immediateStop();
  } else if (mode == 'R') {
    fallEstopLatched = false;
    commandMode = 'S';
    targetDistance = 0;
    targetCenterX = 0.5f;
    targetDetected = false;
    immediateStop();
  } else if (mode == 'S') {
    commandMode = 'S';
    targetDistance = 0;
    targetCenterX = 0.5f;
    targetDetected = false;
  } else {
    commandMode = 'F';
    targetDistance = (int)distanceValue;
    targetCenterX = centerValue;
    targetDetected = detectedValue == 1;
  }
  return true;
}

void receiveSerial() {
  while (Serial.available() > 0) {
    char received = Serial.read();
    if (received == '<') {
      // A new start marker always re-synchronizes the parser.
      receiveInProgress = true;
      receiveOverflow = false;
      receiveIndex = 0;
    } else if (receiveInProgress && received == '>') {
      if (!receiveOverflow) {
        receivedChars[receiveIndex] = '\0';
        parseData();
      }
      receiveInProgress = false;
      receiveOverflow = false;
      receiveIndex = 0;
    } else if (receiveInProgress) {
      if (receiveIndex < MAX_CHARS - 1) {
        receivedChars[receiveIndex++] = received;
      } else {
        receiveOverflow = true;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Obstacle and following control.
// ---------------------------------------------------------------------------
bool obstacleDanger() {
  return usLeft <= OA_TRIGGER_CM
      || usCenter <= OA_TRIGGER_CM
      || usRight <= OA_TRIGGER_CM;
}

bool obstacleClear() {
  return usLeft > OA_CLEAR_CM
      && usCenter > OA_CLEAR_CM
      && usRight > OA_CLEAR_CM;
}

bool handleObstacleAvoidance(unsigned long now) {
  bool danger = obstacleDanger();
  if (!obstacleActive && danger) {
    obstacleActive = true;
    obstacleDirection = usRight >= usLeft ? 1 : -1;
    obstacleStartedAt = now;
    obstacleClearSince = 0;
    currentState = obstacleDirection > 0
        ? STATE_AVOID_RIGHT : STATE_AVOID_LEFT;
    immediateStop();
  }

  if (!obstacleActive) return false;

  if (danger) {
    obstacleClearSince = 0;
  } else if (obstacleClear()) {
    if (obstacleClearSince == 0) obstacleClearSince = now;
    if (now - obstacleClearSince >= OA_CLEAR_CONFIRM_MS) {
      obstacleActive = false;
      obstacleClearSince = 0;
      setDriveTarget(0, 0);
      return false;
    }
  } else {
    obstacleClearSince = 0;
  }

  if (now - obstacleStartedAt < OA_STOP_DWELL_MS) {
    immediateStop();
  } else if (obstacleDirection > 0) {
    setDriveTarget(SPEED_AVOID, -SPEED_AVOID);
    currentState = STATE_AVOID_RIGHT;
  } else {
    setDriveTarget(-SPEED_AVOID, SPEED_AVOID);
    currentState = STATE_AVOID_LEFT;
  }
  return true;
}

int proportionalPower(float value, float startValue, float maximumValue,
                      int minimumPower, int maximumPower) {
  if (value <= startValue) return minimumPower;
  float span = max(0.001f, maximumValue - startValue);
  float ratio = min(1.0f, (value - startValue) / span);
  return minimumPower + (int)((maximumPower - minimumPower) * ratio);
}

int calculateTurnPID(float error, unsigned long now) {
  float absoluteError = fabs(error);
  if (absoluteError < CX_DEADBAND) {
    turnIntegral = 0.0f;
    lastHorizontalError = error;
    return 0;
  }

  float dt = (lastPidTime > 0) ? (now - lastPidTime) / 1000.0f : 0.05f;
  if (dt <= 0.0f || dt > 0.5f) dt = 0.05f;
  lastPidTime = now;

  // Proportional term
  float pTerm = TURN_KP * error;

  // Integral term with Anti-Windup clamping
  turnIntegral += error * dt;
  if (turnIntegral > TURN_INTEGRAL_LIMIT) turnIntegral = TURN_INTEGRAL_LIMIT;
  else if (turnIntegral < -TURN_INTEGRAL_LIMIT) turnIntegral = -TURN_INTEGRAL_LIMIT;
  float iTerm = TURN_KI * turnIntegral;

  // Derivative term (Braking damping against angular overshoot)
  float derivative = (error - lastHorizontalError) / dt;
  lastHorizontalError = error;
  float dTerm = TURN_KD * derivative;

  float pidOutput = pTerm + iTerm + dTerm;
  float mag = fabs(pidOutput);
  if (mag < 5.0f) return 0;

  // Smooth mapping to [SPEED_TURN_MIN, SPEED_TURN_MAX] with friction breakaway
  int turnPower = SPEED_TURN_MIN + (int)((SPEED_TURN_MAX - SPEED_TURN_MIN) * min(1.0f, mag / 80.0f));
  turnPower = min(SPEED_TURN_MAX, max(SPEED_TURN_MIN, turnPower));

  return pidOutput >= 0.0f ? turnPower : -turnPower;
}

void updateFollowingControl(unsigned long now) {
  if (targetDistance <= DIST_TOO_CLOSE) {
    // No rear sonar: stop rather than reversing blind.
    currentState = STATE_TOO_CLOSE;
    forwardActive = false;
    turnIntegral = 0.0f;
    immediateStop();
    return;
  }

  if (forwardActive) {
    if (targetDistance <= DIST_FORWARD_STOP) forwardActive = false;
  } else if (targetDistance >= DIST_FORWARD_START) {
    forwardActive = true;
  }

  float horizontalError = targetCenterX - 0.5f;
  int signedTurn = calculateTurnPID(horizontalError, now);
  turnActive = (signedTurn != 0);

  // Detect rising edge of forwardActive to trigger feedforward kickstart
  if (forwardActive && !prevForwardActive) {
    forwardStartedAt = now;
  }
  prevForwardActive = forwardActive;

  int linearPower = 0;
  if (forwardActive) {
    // 3. Stiction Feedforward Kickstart:
    // When starting motion from rest with a 1.7 kg chassis, provide a brief feedforward boost
    // to instantly overcome static friction without waiting for PWM ramping.
    if (now - forwardStartedAt < KICKSTART_FEEDFORWARD_DURATION_MS) {
      linearPower = KICKSTART_FEEDFORWARD_PWM;
    } else {
      linearPower = proportionalPower(
          (float)targetDistance,
          (float)DIST_FORWARD_START,
          (float)DIST_ESTIMATOR_MAX,
          SPEED_FORWARD_MIN,
          SPEED_FORWARD_MAX);
    }
  }

  // Large heading errors pivot in place slowly; smaller errors create gentle forward arc
  if (turnActive && fabs(horizontalError) >= CX_PIVOT_ERROR) {
    linearPower = 0;
  } else if (linearPower > 0 && abs(signedTurn) > (linearPower * 2) / 3) {
    signedTurn = (signedTurn > 0) ? (linearPower * 2) / 3 : -(linearPower * 2) / 3;
  }

  // 4. Asymmetric Arc Turn Power Allocation:
  // When forward moving (arc turn), avoid dropping inner wheel below minimum drive power
  // (which would cause the 1.7 kg chassis to drag and stall). Instead, boost outer wheel!
  int leftPower = 0;
  int rightPower = 0;
  if (linearPower > 0 && turnActive) {
    if (signedTurn > 0) {
      // Turn Right: left wheels drive faster, right wheels maintain at least base forward speed
      leftPower = clampPower(linearPower + signedTurn);
      rightPower = max(SPEED_FORWARD_MIN, linearPower - (signedTurn / 2));
    } else {
      // Turn Left: right wheels drive faster, left wheels maintain at least base forward speed
      leftPower = max(SPEED_FORWARD_MIN, linearPower - (abs(signedTurn) / 2));
      rightPower = clampPower(linearPower + abs(signedTurn));
    }
  } else {
    leftPower = clampPower(linearPower + signedTurn);
    rightPower = clampPower(linearPower - signedTurn);
  }
  setDriveTarget(leftPower, rightPower);

  if (linearPower == 0 && signedTurn == 0) {
    currentState = STATE_IN_RANGE;
  } else if (linearPower == 0 && signedTurn < 0) {
    currentState = STATE_TURN_LEFT;
  } else if (linearPower == 0 && signedTurn > 0) {
    currentState = STATE_TURN_RIGHT;
  } else if (signedTurn < 0) {
    currentState = STATE_ARC_LEFT;
  } else if (signedTurn > 0) {
    currentState = STATE_ARC_RIGHT;
  } else {
    currentState = STATE_FORWARD;
  }
}

void updateStateMachine(unsigned long now) {
  if (now - lastDataTime > SERIAL_TIMEOUT_MS) {
    currentState = STATE_LINK_ESTOP;
    resetMotionIntent();
    immediateStop();
    return;
  }
  if (fallEstopLatched) {
    currentState = STATE_FALL_ESTOP;
    resetMotionIntent();
    immediateStop();
    return;
  }
  if (commandMode != 'F') {
    currentState = STATE_DISABLED;
    resetMotionIntent();
    immediateStop();
    return;
  }
  if (!targetDetected || targetDistance <= 0) {
    currentState = STATE_IDLE;
    resetMotionIntent();
    immediateStop();
    return;
  }
  if (handleObstacleAvoidance(now)) return;
  updateFollowingControl(now);
}

const char *stateName(RobotState state) {
  switch (state) {
    case STATE_IDLE: return "IDLE";
    case STATE_DISABLED: return "DISABLED";
    case STATE_FORWARD: return "FWD";
    case STATE_ARC_LEFT: return "ARC_L";
    case STATE_ARC_RIGHT: return "ARC_R";
    case STATE_TURN_LEFT: return "TURN_L";
    case STATE_TURN_RIGHT: return "TURN_R";
    case STATE_IN_RANGE: return "IN_RANGE";
    case STATE_TOO_CLOSE: return "TOO_CLOSE";
    case STATE_AVOID_LEFT: return "AVOID_L";
    case STATE_AVOID_RIGHT: return "AVOID_R";
    case STATE_FALL_ESTOP: return "FALL_ESTOP";
    case STATE_LINK_ESTOP: return "LINK_ESTOP";
    default: return "UNKNOWN";
  }
}

void sendTelemetry() {
  Serial.print("S,");
  Serial.print(stateName(currentState));
  Serial.print(",");
  Serial.print(usLeft);
  Serial.print(",");
  Serial.print(usCenter);
  Serial.print(",");
  Serial.print(usRight);
  Serial.print(",");
  Serial.println(fallEstopLatched ? 1 : 0);
}

void setup() {
  Serial.begin(115200);

  pinMode(ENA_L, OUTPUT); pinMode(ENB_L, OUTPUT);
  pinMode(IN1_LF, OUTPUT); pinMode(IN2_LF, OUTPUT);
  pinMode(IN3_LR, OUTPUT); pinMode(IN4_LR, OUTPUT);
  pinMode(ENA_R, OUTPUT); pinMode(ENB_R, OUTPUT);
  pinMode(IN1_RF, OUTPUT); pinMode(IN2_RF, OUTPUT);
  pinMode(IN3_RR, OUTPUT); pinMode(IN4_RR, OUTPUT);

  pinMode(TRIG_LEFT, OUTPUT); pinMode(ECHO_LEFT, INPUT);
  pinMode(TRIG_CENTER, OUTPUT); pinMode(ECHO_CENTER, INPUT);
  pinMode(TRIG_RIGHT, OUTPUT); pinMode(ECHO_RIGHT, INPUT);

  immediateStop();
  lastDataTime = millis();
  Serial.println("# HAFIBOT Motor Controller v4.0");
  Serial.println("# Protocol v2; following disabled until commanded");
}

void loop() {
  unsigned long now = millis();
  receiveSerial();

  if (now - lastUSReadTime >= US_READ_INTERVAL_MS) {
    lastUSReadTime = now;
    updateUltrasonic();
  }
  if (now - lastControlTime >= CONTROL_INTERVAL_MS) {
    lastControlTime = now;
    updateStateMachine(now);
  }
  if (now - lastMotorRampTime >= MOTOR_RAMP_INTERVAL_MS) {
    lastMotorRampTime = now;
    serviceMotorRamp();
  }
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = now;
    sendTelemetry();
  }
}
