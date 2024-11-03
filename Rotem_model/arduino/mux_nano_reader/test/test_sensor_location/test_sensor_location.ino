//16-Channel MUX (74HC4067) Interface
//===================================
int S[4] = {14, 15, 16, 17};
int A[4] = {A4, A5, A6, A7}; // If only two channels are used
int sensor_list[32];
int loc = 0;

bool MUXtable[16][4] = {
  {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0},
  {0, 0, 1, 0}, {1, 0, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 0},
  {0, 0, 0, 1}, {1, 0, 0, 1}, {0, 1, 0, 1}, {1, 1, 0, 1},
  {0, 0, 1, 1}, {1, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 1}
};


void setup() {
    for (int i = 0; i < 4; i++) {
    pinMode(S[i], OUTPUT);
    digitalWrite(S[i], LOW);
  }
  Serial.begin(115200);
  Serial.println("hello");
//  analogReadResolution(10);

}

void loop() {
  String dataString = "";

    for (int i = 0; i < 16; i++) {
    selection(i);
    //delayMicroseconds(100); // Adjust as necessary
    for (int j = 0; j < 2; j++) {
      if (analogRead(A[j])>100)
      {

        Serial.println(j*(16)+i); 
      }
    }
  }
}


void selection(int j) {
  for (int k = 0; k < 4; k++) {
    digitalWrite(S[k], MUXtable[j][k]);
  }
}
