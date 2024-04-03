void setup() {
  // Initialize serial communication at 115200 baud rate
  Serial.begin(115200);
}

void loop() {
  // Send data to the serial port
  Serial.println("Hello, world!");

  // Wait for a second
  delay(1000);
}
