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
const int SPEED_FORWARD_MIN = 145;  // Gentle rolling floor (smooth, quiet forward roll)
const int SPEED_FORWARD_MAX = 185;  // Calm cruising ceiling (no aggressive surging)
const int SPEED_STRAFE_MIN = 145;   // Boosted lateral floor to break 45-deg roller stiction without stall
const int SPEED_STRAFE_MAX = 170;   // Controlled lateral ceiling (~90% forward max) for smooth sideways slide
const int SPEED_TURN_MIN = 65;      // Soft initial steering torque
const int SPEED_TURN_MAX = 110;     // Relaxed maximum turn rate to prevent FOV target loss
const int SPEED_AVOID = 130;        // Gentle obstacle avoidance pivot speed
const int PWM_RAMP_STEP = 6;        // Silky smooth acceleration ramp (gradual throttle)
const int KICKSTART_FEEDFORWARD_PWM = 175; // Soft breakaway pulse (eliminates start jerk)
const unsigned long KICKSTART_FEEDFORWARD_DURATION_MS = 60; // Brief 60 ms stiction breakaway duration

// Mecanum Holonomic Geometry & Kinematic Configurations:
// Invert STRAFE_SIGN (-1) if lateral strafing moves opposite to target direction.
const int STRAFE_SIGN = 1;
// Geometry factor: (wheelbase_L + track_width_W) / 2 scale factor for yaw rotation (calibrate per chassis)
const float MECANUM_GEOMETRY_FACTOR = 1.0f;

// Human following distance parameters:
// Following active when user > 200 cm: standby zone 150-200 cm, stops approaching at 150 cm,
// and safety stops if user is closer than 100 cm.
const int DIST_TOO_CLOSE = 100;
const int DIST_FORWARD_STOP = 150;
const int DIST_FORWARD_START = 200;
const int DIST_ESTIMATOR_MAX = 500;

// Heading Alignment PID Controller Constants:
const float TURN_KP = 60.0f;        // Gentle proportional steering
const float TURN_KI = 0.0f;         // Zero integral to eliminate windup overshoot
const float TURN_KD = 10.0f;        // Gentle derivative damping against overshoot
const float TURN_INTEGRAL_LIMIT = 0.0f;
const float CX_DEADBAND = 0.080f;   // Centered window (0.42 - 0.58, 102px comfort zone)
const float CX_PIVOT_ERROR = 0.25f; // Beyond 0.25 offset: gentle pivot to re-center first

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
  STATE_STRAFE_LEFT,
  STATE_STRAFE_RIGHT,
  STATE_DIAG_LEFT,
  STATE_DIAG_RIGHT,
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
int targetFL = 0;
int targetFR = 0;
int targetRL = 0;
int targetRR = 0;

int currentFL = 0;
int currentFR = 0;
int currentRL = 0;
int currentRR = 0;

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

void writeMotorFL(int signedPower) {
  int power = abs(clampPower(signedPower));
  analogWrite(ENA_L, power);
  if (signedPower > 0) {
    digitalWrite(IN1_LF, HIGH); digitalWrite(IN2_LF, LOW);
  } else if (signedPower < 0) {
    digitalWrite(IN1_LF, LOW); digitalWrite(IN2_LF, HIGH);
  } else {
    digitalWrite(IN1_LF, LOW); digitalWrite(IN2_LF, LOW);
  }
}

void writeMotorRL(int signedPower) {
  int power = abs(clampPower(signedPower));
  analogWrite(ENB_L, power);
  if (signedPower > 0) {
    digitalWrite(IN3_LR, HIGH); digitalWrite(IN4_LR, LOW);
  } else if (signedPower < 0) {
    digitalWrite(IN3_LR, LOW); digitalWrite(IN4_LR, HIGH);
  } else {
    digitalWrite(IN3_LR, LOW); digitalWrite(IN4_LR, LOW);
  }
}

void writeMotorFR(int signedPower) {
  int power = abs(clampPower(signedPower));
  analogWrite(ENA_R, power);
  if (signedPower > 0) {
    digitalWrite(IN1_RF, HIGH); digitalWrite(IN2_RF, LOW);
  } else if (signedPower < 0) {
    digitalWrite(IN1_RF, LOW); digitalWrite(IN2_RF, HIGH);
  } else {
    digitalWrite(IN1_RF, LOW); digitalWrite(IN2_RF, LOW);
  }
}

void writeMotorRR(int signedPower) {
  int power = abs(clampPower(signedPower));
  analogWrite(ENB_R, power);
  if (signedPower > 0) {
    digitalWrite(IN3_RR, HIGH); digitalWrite(IN4_RR, LOW);
  } else if (signedPower < 0) {
    digitalWrite(IN3_RR, LOW); digitalWrite(IN4_RR, HIGH);
  } else {
    digitalWrite(IN3_RR, LOW); digitalWrite(IN4_RR, LOW);
  }
}

void writeMecanumMotors(int fl, int fr, int rl, int rr) {
  writeMotorFL(fl);
  writeMotorFR(fr);
  writeMotorRL(rl);
  writeMotorRR(rr);
}

void setMecanumDriveTarget(int fl, int fr, int rl, int rr) {
  targetFL = clampPower(fl);
  targetFR = clampPower(fr);
  targetRL = clampPower(rl);
  targetRR = clampPower(rr);
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
  setMecanumDriveTarget(leftPower, rightPower, leftPower, rightPower);
}

void immediateStop() {
  targetFL = targetFR = targetRL = targetRR = 0;
  currentFL = currentFR = currentRL = currentRR = 0;
  writeMecanumMotors(0, 0, 0, 0);
}

int rampPower(int currentPower, int requestedPower) {
  // Active dynamic deceleration when requested to stop
  if (requestedPower == 0) {
    if (abs(currentPower) <= PWM_RAMP_STEP * 2) return 0;
    return (currentPower > 0) ? currentPower - PWM_RAMP_STEP * 2 : currentPower + PWM_RAMP_STEP * 2;
  }

  // Always pass through zero before reversing direction
  if (currentPower > 0 && requestedPower < 0) {
    return max(0, currentPower - PWM_RAMP_STEP * 2);
  }
  if (currentPower < 0 && requestedPower > 0) {
    return min(0, currentPower + PWM_RAMP_STEP * 2);
  }

  // Starting from complete stop: jump straight to requestedPower
  if (currentPower == 0) {
    return requestedPower;
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
  currentFL = rampPower(currentFL, targetFL);
  currentFR = rampPower(currentFR, targetFR);
  currentRL = rampPower(currentRL, targetRL);
  currentRR = rampPower(currentRR, targetRR);
  writeMecanumMotors(currentFL, currentFR, currentRL, currentRR);
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
  if (mag < 2.0f) return 0;

  // Smooth mapping to [SPEED_TURN_MIN, SPEED_TURN_MAX] with friction breakaway
  int turnPower = SPEED_TURN_MIN + (int)((SPEED_TURN_MAX - SPEED_TURN_MIN) * min(1.0f, mag / 30.0f));
  turnPower = min(SPEED_TURN_MAX, max(SPEED_TURN_MIN, turnPower));

  return pidOutput >= 0.0f ? turnPower : -turnPower;
}

void computeMecanumKinematics(int vx, int vy, int omega) {
  // Mecanum inverse kinematics for 4-wheel independent drive:
  float fl = (float)vx + (float)(STRAFE_SIGN * vy) + (float)omega * MECANUM_GEOMETRY_FACTOR;
  float fr = (float)vx - (float)(STRAFE_SIGN * vy) - (float)omega * MECANUM_GEOMETRY_FACTOR;
  float rl = (float)vx - (float)(STRAFE_SIGN * vy) + (float)omega * MECANUM_GEOMETRY_FACTOR;
  float rr = (float)vx + (float)(STRAFE_SIGN * vy) - (float)omega * MECANUM_GEOMETRY_FACTOR;

  // Proportional vector normalization: preserves vx:vy:omega velocity ratio if any wheel > 255
  float maxVal = max(max(fabs(fl), fabs(fr)), max(fabs(rl), fabs(rr)));
  if (maxVal > 255.0f) {
    float scale = 255.0f / maxVal;
    fl *= scale;
    fr *= scale;
    rl *= scale;
    rr *= scale;
  }

  setMecanumDriveTarget((int)round(fl), (int)round(fr), (int)round(rl), (int)round(rr));
}

void updateFollowingControl(unsigned long now) {
  // Rule 1: Distance too close safety interlock -> All speeds 0
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
  float absError = fabs(horizontalError);

  // Detect rising edge of forwardActive to trigger feedforward kickstart
  if (forwardActive && !prevForwardActive) {
    forwardStartedAt = now;
  }
  prevForwardActive = forwardActive;

  int vx = 0;
  if (forwardActive) {
    // 3. Stiction Feedforward Kickstart:
    // When starting motion from rest with a 1.7 kg chassis, provide a brief feedforward boost
    // to instantly overcome static friction without waiting for PWM ramping.
    if (now - forwardStartedAt < KICKSTART_FEEDFORWARD_DURATION_MS) {
      vx = KICKSTART_FEEDFORWARD_PWM;
    } else {
      vx = proportionalPower(
          (float)targetDistance,
          (float)DIST_FORWARD_STOP,
          (float)DIST_ESTIMATOR_MAX,
          SPEED_FORWARD_MIN,
          SPEED_FORWARD_MAX);
    }
  }

  int vy = 0;
  int omega = 0;

  // Rule 2a: Deadband -> vy = 0, omega = 0. Pure longitudinal motion.
  if (absError <= CX_DEADBAND) {
    vy = 0;
    omega = 0;
    currentState = (vx > 0) ? STATE_FORWARD : STATE_IN_RANGE;
  }
  // Rule 2b: Holonomic Strafing (CX_DEADBAND < |error| < CX_PIVOT_ERROR)
  else if (absError < CX_PIVOT_ERROR) {
    float spanError = max(0.001f, CX_PIVOT_ERROR - CX_DEADBAND);
    float strafeRatio = min(1.0f, max(0.0f, (absError - CX_DEADBAND) / spanError));
    int strafeSpeed = SPEED_STRAFE_MIN + (int)((SPEED_STRAFE_MAX - SPEED_STRAFE_MIN) * strafeRatio);
    vy = (horizontalError > 0.0f) ? strafeSpeed : -strafeSpeed;
    omega = 0;

    if (vx > 0) {
      currentState = (horizontalError > 0.0f) ? STATE_DIAG_RIGHT : STATE_DIAG_LEFT;
    } else {
      currentState = (horizontalError > 0.0f) ? STATE_STRAFE_RIGHT : STATE_STRAFE_LEFT;
    }
  }
  // Rule 2c: Large offset (|error| >= CX_PIVOT_ERROR, default 0.25) -> Arc/Pivot
  else {
    vy = 0;
    omega = calculateTurnPID(horizontalError, now);
    vx = 0; // Pivot in place to center person first
    currentState = (omega > 0) ? STATE_TURN_RIGHT : STATE_TURN_LEFT;
  }

  computeMecanumKinematics(vx, vy, omega);
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
    case STATE_STRAFE_LEFT: return "STRF_L";
    case STATE_STRAFE_RIGHT: return "STRF_R";
    case STATE_DIAG_LEFT: return "DIAG_L";
    case STATE_DIAG_RIGHT: return "DIAG_R";
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
