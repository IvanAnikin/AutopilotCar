
#include <SoftwareSerial.h>
#include <Servo.h>

SoftwareSerial mySerial(10, 11); // RX, TX - parners

Servo servo_lower;
Servo servo_upper;

// VARIABLES NEEDED FOR THE CODE
float Speed = 255;  // out of 255 max
int forwardStep = 20;
int rotationStep = 20;
int smallForwardStep = 10;
int smallRotationStep = 10;
String message;
int movement;

// SERVOS
const int max_lower = 160;
const int min_lower = 20;
const int max_upper = 140;
const int min_upper = 115;
const int lower_step = 1;
const int upper_step = 4;

int pos_lower = 75;
int pos_upper = 130; //min_upper

// DISTANCE SENSOR
#define trig 4
#define echo 5
int distance = 0;
long previousMillis = 0;
long command_max_interval = 5000;

// Motors sensors
int left_intr = 0;
int right_intr = 0;
int angle = 0;
float radius_of_wheel = 0.033;  //Measure the radius of your wheel and enter it here in cm
volatile byte rotation; // variale for interrupt fun must be volatile
float timetaken,rpm,dtime;
float v;
int distance_motors;
unsigned long pevtime;
bool left_smaller = true;
bool right_smaller = true;
String left_smaller_t = "";
String right_smaller_t = "";
int wheels_position = 0;


// PINS NUMBERS DECLARATION
int in1 = 52;
int in2 = 50;
int in3 = 48;
int in4 = 46;
int Btn = 44;
int rx_distance = 18;
int tx_distance = 19;
int servo_lower_pin = 8;
int servo_upper_pin = 9;
int en1 = 6;
int en2 = 7;

void setup() {
  Serial.begin(9600);     // Serial communication using cable
  mySerial.begin(9600);   // Serial communication with esp32 module

  // WHEELS PINS
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(en1, OUTPUT);
  pinMode(en2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  analogWrite(en1, Speed);
  analogWrite(en2, Speed);

  // SERVOS PINS
  servo_lower.attach(servo_lower_pin);
  servo_upper.attach(servo_upper_pin);

  servo_upper.write(pos_upper);
  servo_lower.write(pos_lower);

  // DISTANCE PINS
  pinMode(trig, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echo, INPUT);

  // Motors sensors setup
  rotation = rpm = pevtime = 0; //Initialize all variable to zero
  attachInterrupt(digitalPinToInterrupt(2), Left_ISR, CHANGE); //Left_ISR is called when left wheel sensor is triggered
  attachInterrupt(digitalPinToInterrupt(3), Right_ISR, CHANGE);//Right_ISR is called when right wheel sensor is triggered

  Serial.println("Setup ended");
}

void loop() {
  //distance = getDistance();
  //Serial.println("Distance: " + String(distance));
  //mySerial.println(String(distance));
  //delay(500);
  if(mySerial.available()){
    message = mySerial.readStringUntil('\n'); // Message format: "*" - data type, ":", "*n" - n-th count of data (movement: 1)
    Serial.println(message);
    if(message.substring(0, 1) == "M"){       // // "M" MOVEMENT: data count: 1 {0,1,2,3} - {forward, backward, left, right}
      movement = message.substring(2,3).toInt();

      if(movement == 0) Go_d(forwardStep);
      //if(movement == 1) Back();
      if(movement == 2) Left_d(rotationStep);
      if(movement == 3) Right_d(rotationStep);

      delay(100);
      Stop();
    }
    if(message.substring(0, 1) == "S"){       // "S" SERVOS: data count: 1 {0,1,2,3,4} - {up, down, left, right, default}
      movement = message.substring(2,3).toInt();

      if(movement == 0) pos_upper += 10;
      if(movement == 1) pos_upper -= 10;
      if(movement == 2) pos_lower += 10;
      if(movement == 3) pos_lower -= 10;
      if(movement == 4) ServosDefault();

      servo_upper.write(pos_upper);
      servo_lower.write(pos_lower);
    }
    if(message.substring(0, 1) == "H"){       // // "H" HALF-STEP: data count: 1 {0,1,2,3} - {forward, backward, left, right}
      movement = message.substring(2,3).toInt();

      if(movement == 0) Go_d(smallForwardStep);
      //if(movement == 1) Back();
      if(movement == 2) Left_d(smallRotationStep);
      if(movement == 3) Right_d(smallRotationStep);

      delay(100);
      Stop();
    }


    // "R" SPEED: data count: 1 {0,1,2,3} - {slowest->fastest}

    // Send distance through Esp32 module to computer after finishing the command
    distance = getDistance();
    Serial.println("Distance: " + String(distance));
    mySerial.println(String(distance));
    //Serial.println(distance);
  }

  // Motors sensors
  /*To drop to zero if vehicle stopped*/
  if(millis()-dtime>500) //no inetrrupt found for 500ms
  {
    rpm = v = 0; // make rpm and velocity as zero
    dtime = millis();
  }

  delay(100); // 30/50
}




void Left_ISR()
{
  left_intr++;delay(10);
}

void Right_ISR()
{
  right_intr++; delay(10);

  rotation++;
  dtime=millis();
  if(rotation>=40)
  {
    timetaken = millis()-pevtime; //timetaken in millisec
    rpm=(1000/timetaken)*60;    //formulae to calculate rpm
    pevtime = millis();
    rotation=0;
  }
}
int getDistance(){
  digitalWrite(trig, LOW);
  delayMicroseconds(2);

  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  distance = pulseIn(echo, HIGH);
  distance = distance / 58;

  return distance;
}

void ServosDefault(){
  pos_lower = 70;
  pos_upper = min_upper;
  servo_upper.write(pos_upper);
  servo_lower.write(pos_lower);
}
void Go(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}
void Back(){
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void Right(){
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}
void Left(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void Stop(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}
void Go_d(int step){
  wheels_position += step;
  left_smaller = right_smaller = true;
  previousMillis = millis();
  while((left_smaller or right_smaller) and (millis() - previousMillis < command_max_interval)){
    if(right_smaller){
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
    }else{
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
    }if(left_smaller){
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
    }else{
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
    }
    if(left_smaller) left_smaller_t = "True";
    else left_smaller_t = "False";
    if(right_smaller) right_smaller_t = "True";
    else right_smaller_t = "False";
    //Serial.println("left_intr: '" + String(left_intr) + "' right_intr: '" + String(right_intr) + "' left_smaller: '" + left_smaller_t + "' right_smaller: '" + right_smaller_t + "' wheels_position: '" + String(wheels_position) + "'");
    delay(5);
    left_smaller = left_intr < wheels_position;
    right_smaller = right_intr < wheels_position;
  }
  Serial.println("Go done");
}
void Right_d(int step){
  wheels_position += step;
  left_smaller = right_smaller = true;
  previousMillis = millis();
  while((left_smaller or right_smaller) and (millis() - previousMillis < command_max_interval)){
    if(right_smaller){
      digitalWrite(in1, HIGH);
      digitalWrite(in2, LOW);
    }else{
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
    }if(left_smaller){
      digitalWrite(in3, LOW);
      digitalWrite(in4, HIGH);
    }else{
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
    }
    if(left_smaller) left_smaller_t = "True";
    else left_smaller_t = "False";
    if(right_smaller) right_smaller_t = "True";
    else right_smaller_t = "False";
    //Serial.println("left_intr: '" + String(left_intr) + "' right_intr: '" + String(right_intr) + "' left_smaller: '" + left_smaller_t + "' right_smaller: '" + right_smaller_t + "' wheels_position: '" + String(wheels_position) + "'");
    delay(5);
    left_smaller = left_intr < wheels_position;
    right_smaller = right_intr < wheels_position;
  }
  Serial.println("Right done");
}
void Left_d(int step){
  wheels_position += step;
  left_smaller = right_smaller = true;
  previousMillis = millis();
  while((left_smaller or right_smaller) and (millis() - previousMillis < command_max_interval)){
    if(right_smaller){
      digitalWrite(in1, LOW);
      digitalWrite(in2, HIGH);
    }else{
      digitalWrite(in1, LOW);
      digitalWrite(in2, LOW);
    }if(left_smaller){
      digitalWrite(in3, HIGH);
      digitalWrite(in4, LOW);
    }else{
      digitalWrite(in3, LOW);
      digitalWrite(in4, LOW);
    }
    if(left_smaller) left_smaller_t = "True";
    else left_smaller_t = "False";
    if(right_smaller) right_smaller_t = "True";
    else right_smaller_t = "False";
    //Serial.println("left_intr: '" + String(left_intr) + "' right_intr: '" + String(right_intr) + "' left_smaller: '" + left_smaller_t + "' right_smaller: '" + right_smaller_t + "' wheels_position: '" + String(wheels_position) + "'");
    delay(5);
    left_smaller = left_intr < wheels_position;
    right_smaller = right_intr < wheels_position;
  }
  Serial.println("Left done");
}