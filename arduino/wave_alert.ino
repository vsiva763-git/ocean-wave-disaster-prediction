// Arduino sketch for displaying wave risk and sending alerts.
// Hardware: LCD 16x2 (I2C), buzzer, GSM module (SIM800L), optional LED.

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>

// Pin configuration
const int BUZZER_PIN = 9;
const int LED_PIN = 8;

// Replace with your SIM800L TX/RX wiring
const int GSM_RX = 10; // SIM800L TX -> Arduino RX
const int GSM_TX = 11; // SIM800L RX -> Arduino TX

LiquidCrystal_I2C lcd(0x27, 16, 2);
SoftwareSerial gsm(GSM_RX, GSM_TX);

String phoneNumber = "+1234567890"; // TODO: set your phone number

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  gsm.begin(9600);

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Wave Alert Ready");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    // Expected format: LABEL:GIANT;HPI:0.87;P_GIANT:0.92
    String label = parseValue(line, "LABEL");
    String hpi = parseValue(line, "HPI");
    String pGiant = parseValue(line, "P_GIANT");

    displayAlert(label, hpi, pGiant);
    handleAlert(label, pGiant);
  }
}

String parseValue(const String &src, const String &key) {
  int idx = src.indexOf(key + ":");
  if (idx == -1) return "";
  int start = idx + key.length() + 1;
  int end = src.indexOf(';', start);
  if (end == -1) end = src.length();
  return src.substring(start, end);
}

void displayAlert(const String &label, const String &hpi, const String &pGiant) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Risk: " + label);
  lcd.setCursor(0, 1);
  lcd.print("HPI:" + hpi + " PG:" + pGiant);
}

void handleAlert(const String &label, const String &pGiant) {
  bool giant = label == "GIANT";
  if (giant) {
    tone(BUZZER_PIN, 2000, 800);
    digitalWrite(LED_PIN, HIGH);
    sendSMS(label, pGiant);
  } else if (label == "MODERATE") {
    tone(BUZZER_PIN, 1200, 400);
    digitalWrite(LED_PIN, HIGH);
    delay(300);
    digitalWrite(LED_PIN, LOW);
  } else {
    noTone(BUZZER_PIN);
    digitalWrite(LED_PIN, LOW);
  }
}

void sendSMS(const String &label, const String &pGiant) {
  gsm.print("AT+CMGF=1\r");
  delay(500);
  gsm.print("AT+CMGS=\"" + phoneNumber + "\"\r");
  delay(500);
  gsm.print("GIANT WAVE ALERT\n");
  gsm.print("Label: " + label + "\nP(Giant): " + pGiant + "\nTake immediate action.\r");
  gsm.write(26); // CTRL+Z to send
  delay(1000);
}
