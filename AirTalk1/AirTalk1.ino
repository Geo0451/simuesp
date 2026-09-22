#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#define MPU_ADDR         0x68
#define TOUCH_PIN        27
#define LED_PIN          2

// 29-byte packed binary packet structure (zero padding)
struct __attribute__((packed)) TelemetryPacket {
  uint32_t timestampUs;
  uint8_t touched;
  float ax, ay, az;
  float gx, gy, gz;
};

QueueHandle_t telemetryQueue;
WiFiUDP udp;
const IPAddress broadcastIP(192, 168, 4, 255); // UDP broadcast over SoftAP
const uint16_t udpPort = 5005;

float aBias[3] = {0}, gBias[3] = {0};
bool touchState = false;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

void readRawIMU(int16_t raw[6]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
  
  for (int i = 0; i < 3; i++) raw[i] = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read(); // Skip temp bytes
  for (int i = 3; i < 6; i++) raw[i] = (Wire.read() << 8) | Wire.read();
}

void calibrate() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("[IMU] Calibrating biases...");
  delay(1000);

  double sA[3] = {0}, sG[3] = {0};
  uint32_t count = 0;
  uint32_t start = millis();

  while (millis() - start < 2000 || count < 800) {
    int16_t r[6];
    readRawIMU(r);
    for (int i = 0; i < 3; i++) {
      sA[i] += r[i] / 16384.0f;
      sG[i] += r[i + 3] / 131.0f;
    }
    count++;
    delay(2);
  }

  for (int i = 0; i < 3; i++) {
    aBias[i] = sA[i] / count;
    gBias[i] = sG[i] / count;
  }
  aBias[2] -= 1.0f; // Subtract 1G gravity on Z

  digitalWrite(LED_PIN, LOW);
  Serial.printf("[IMU] Calibrated. aBias:[%.3f,%.3f,%.3f] gBias:[%.3f,%.3f,%.3f]\n",
                aBias[0], aBias[1], aBias[2], gBias[0], gBias[1], gBias[2]);
}

bool readTouch() {
  static bool cand = false;
  static uint8_t cnt = 0;
  bool raw = (digitalRead(TOUCH_PIN) == HIGH);
  if (raw == cand) {
    if (cnt < 2) cnt++;
  } else {
    cand = raw;
    cnt = 1;
  }
  if (cnt >= 2) touchState = cand;
  return touchState;
}

// -------------------------------------------------------------------
// CORE 1 TASK: High-Speed IMU & Touch Sampling (100 Hz)
// -------------------------------------------------------------------
void imuTask(void *pvParameters) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = pdMS_TO_TICKS(10); // 10ms = 100Hz

  for (;;) {
    vTaskDelayUntil(&xLastWakeTime, xFrequency);

    int16_t r[6];
    readRawIMU(r);

    TelemetryPacket pkt;
    pkt.timestampUs = micros();
    pkt.touched = readTouch() ? 1 : 0;

    // Physical unit conversions (g-force & deg/s) minus static bias
    pkt.ax = (r[0] / 16384.0f) - aBias[0];
    pkt.ay = (r[1] / 16384.0f) - aBias[1];
    pkt.az = (r[2] / 16384.0f) - aBias[2];
    
    pkt.gx = (r[3] / 131.0f) - gBias[0];
    pkt.gy = (r[4] / 131.0f) - gBias[1];
    pkt.gz = (r[5] / 131.0f) - gBias[2];

    xQueueSend(telemetryQueue, &pkt, 0);
  }
}

// -------------------------------------------------------------------
// CORE 0 TASK: SoftAP Network Operations & UDP Binary Streaming
// -------------------------------------------------------------------
void netTask(void *pvParameters) {
  TelemetryPacket pkt;

  for (;;) {
    if (xQueueReceive(telemetryQueue, &pkt, pdMS_TO_TICKS(10)) == pdTRUE) {
      udp.beginPacket(broadcastIP, udpPort);
      udp.write((const uint8_t*)&pkt, sizeof(TelemetryPacket));
      udp.endPacket();
    }
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(TOUCH_PIN, INPUT_PULLDOWN);

  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000); // 400 kHz Fast I2C

  writeReg(0x6B, 0x00); delay(50); // Wake MPU6050
  writeReg(0x6B, 0x01);            // Gyro PLL Clock
  writeReg(0x1A, 0x03);            // DLPF ~42Hz Hardware LPF
  writeReg(0x1B, 0x00);            // Gyro ±250 dps
  writeReg(0x1C, 0x00);            // Accel ±2g

  uint8_t who = readReg(0x75);
  if (who != 0x68 && who != 0x70) {
    while (1) { digitalWrite(LED_PIN, !digitalRead(LED_PIN)); delay(100); }
  }

  calibrate();

  telemetryQueue = xQueueCreate(20, sizeof(TelemetryPacket));

  // Configure ESP32 SoftAP Mode
  WiFi.mode(WIFI_AP);
  WiFi.softAP("AirStylus_AP", "stylus1234");
  Serial.print("[SoftAP] Started AP. Gateway IP: ");
  Serial.println(WiFi.softAPIP());

  udp.begin(udpPort);

  // Pin sampling to Core 1 and network streaming to Core 0
  xTaskCreatePinnedToCore(imuTask, "IMU_Task", 4096, NULL, 2, NULL, 1);
  xTaskCreatePinnedToCore(netTask, "NET_Task", 4096, NULL, 1, NULL, 0);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}