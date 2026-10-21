// =================================================================
// 🚚 Otonom Kargo Aracı - L298N PWM ve PID Motor Kontrol Yazılımı
// Pinler: 5, 6, 9, 10 (Arduino Uno Donanımsal PWM Pinleri)
// =================================================================

const int motorA1 = 5;  // Sol Motorlar İleri (PWM)
const int motorA2 = 6;  // Sol Motorlar Geri  (PWM)
const int motorB1 = 9;  // Sağ Motorlar İleri (PWM)
const int motorB2 = 10; // Sağ Motorlar Geri  (PWM)

// Seri İletişim Tamponu (<left_pwm,right_pwm>)
char serialBuffer[32];
byte bufferIndex = 0;
bool receivingPacket = false;

// Güvenlik Zamanlayıcısı (Watchdog Timer)
unsigned long lastCommandTime = 0;
const unsigned long TIMEOUT_MS =
    1500; // 1.5 saniye veri gelmezse motorları otomatik durdur

void setup() {
  // Motor kontrol pinlerini çıkış yap
  pinMode(motorA1, OUTPUT);
  pinMode(motorA2, OUTPUT);
  pinMode(motorB1, OUTPUT);
  pinMode(motorB2, OUTPUT);

  // Başlangıçta motorları tamamen durdur
  dur();

  // Seri iletişimi başlat (Python ile 9600 baud)
  Serial.begin(9600);
  Serial.println("Otonom Kargo Araci PID Modu Hazir.");
  lastCommandTime = millis();
}

void loop() {
  // Python'dan gelen seri paketleri dinle ve işle
  processSerial();

  // Güvenlik: Bağlantı koparsa veya Python donarsa motorları durdur
  if (millis() - lastCommandTime > TIMEOUT_MS) {
    dur();
  }
}

// --- SERİ PAKET AYRIŞTIRMA ---
void processSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    // Paket Başlangıcı
    if (c == '<') {
      bufferIndex = 0;
      receivingPacket = true;
    }
    // Paket Sonu
    else if (c == '>') {
      if (receivingPacket) {
        serialBuffer[bufferIndex] = '\0';
        receivingPacket = false;
        parseAndApplyMotors(serialBuffer);
        lastCommandTime = millis();
      }
    }
    // Veri Karakterlerini Topla
    else if (receivingPacket) {
      if (bufferIndex < sizeof(serialBuffer) - 1) {
        serialBuffer[bufferIndex++] = c;
      }
    }
    // Acil Durdurma Komutu ('B') desteği
    else if (c == 'B') {
      dur();
      lastCommandTime = millis();
    }
  }
}

// "<left_pwm,right_pwm>" metnini tamsayılara dönüştür
void parseAndApplyMotors(char *data) {
  int leftPwm = 0;
  int rightPwm = 0;

  if (sscanf(data, "%d,%d", &leftPwm, &rightPwm) == 2) {
    setMotors(leftPwm, rightPwm);
  }
}

// --- MOTOR SÜRME (DİFERANSİYEL HIZ & PWM) ---
void setMotors(int leftPwm, int rightPwm) {
  // Hızları güvenli sınırlar içinde tut (-255 ile +255 arası)
  leftPwm = constrain(leftPwm, -255, 255);
  rightPwm = constrain(rightPwm, -255, 255);

  // --- SOL MOTORLAR ---
  if (leftPwm > 0) {
    analogWrite(motorA1, leftPwm);
    digitalWrite(motorA2, LOW);
  } else if (leftPwm < 0) {
    digitalWrite(motorA1, LOW);
    analogWrite(motorA2, abs(leftPwm));
  } else {
    digitalWrite(motorA1, LOW);
    digitalWrite(motorA2, LOW);
  }

  // --- SAĞ MOTORLAR ---
  if (rightPwm > 0) {
    analogWrite(motorB1, rightPwm);
    digitalWrite(motorB2, LOW);
  } else if (rightPwm < 0) {
    digitalWrite(motorB1, LOW);
    analogWrite(motorB2, abs(rightPwm));
  } else {
    digitalWrite(motorB1, LOW);
    digitalWrite(motorB2, LOW);
  }
}

void dur() {
  digitalWrite(motorA1, LOW);
  digitalWrite(motorA2, LOW);
  digitalWrite(motorB1, LOW);
  digitalWrite(motorB2, LOW);
}