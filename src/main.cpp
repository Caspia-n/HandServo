#include <ESP32Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;

void setup()
{
  Serial.begin(115200);
  servo1.attach(21); // Index
  servo2.attach(19); // Middle
  servo3.attach(18); // Ring
  servo4.attach(5);  // Pinky
  servo5.attach(4);  // Thumb
}

void loop()
{
  if (Serial.available())
  {
    String data = Serial.readStringUntil('\n');
    int a1, a2, a3, a4, a5;
    if (sscanf(data.c_str(), "%d,%d,%d,%d,%d", &a1, &a2, &a3, &a4, &a5) == 5)
    {
      servo1.write(a1);
      servo2.write(a2);
      servo3.write(a3);
      servo4.write(a4);
      servo5.write(a5);
      Serial.print("Servos set to: ");
      Serial.print(a1);
      Serial.print(", ");
      Serial.print(a2);
      Serial.print(", ");
      Serial.print(a3);
      Serial.print(", ");
      Serial.print(a4);
      Serial.print(", ");
      Serial.println(a5);
    }
    else
    {
      Serial.println("Invalid data");
    }
    delay(5);
  }
}
