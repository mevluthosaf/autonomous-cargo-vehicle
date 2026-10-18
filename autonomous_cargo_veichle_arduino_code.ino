// L298N Motor Sürücü Pin Tanımlamaları
// Raporundaki Şekil 2 ve 5'teki bağlantılara uygun olarak [cite: 95, 115]
const int motorA1 = 5; // Sol Motorlar İleri
const int motorA2 = 6; // Sol Motorlar Geri
const int motorB1 = 9; // Sağ Motorlar İleri
const int motorB2 = 10; // Sağ Motorlar Geri

char komut;

void setup() {
  // Pinleri çıkış olarak ayarla
  pinMode(motorA1, OUTPUT);
  pinMode(motorA2, OUTPUT);
  pinMode(motorB1, OUTPUT);
  pinMode(motorB2, OUTPUT);

  // Seri haberleşmeyi başlat (Python tarafındaki baudrate ile aynı: 9600)
  Serial.begin(9600);
  Serial.println("Otonom Kargo Aracı Hazır.");
}

void loop() {
  if (Serial.available() > 0) {
    komut = Serial.read(); // Python'dan gelen karakteri oku

    if (komut == 'S') { // Düz Git
      ileriGit();
    } 
    else if (komut == 'L') { // Sola Dön
      solaDon();
    } 
    else if (komut == 'R') { // Sağa Dön
      sagaDon();
    } 
    else if (komut == 'B') { // Dur (Fren)
      dur();
    }
  }
}

// --- HAREKET FONKSİYONLARI ---

void ileriGit() {
  digitalWrite(motorA1, HIGH);
  digitalWrite(motorA2, LOW);
  digitalWrite(motorB1, HIGH);
  digitalWrite(motorB2, LOW);
}

void solaDon() {
  digitalWrite(motorA1, LOW);
  digitalWrite(motorA2, HIGH);
  digitalWrite(motorB1, HIGH);
  digitalWrite(motorB2, LOW);
}

void sagaDon() {
  digitalWrite(motorA1, HIGH);
  digitalWrite(motorA2, LOW);
  digitalWrite(motorB1, LOW);
  digitalWrite(motorB2, HIGH);
}

void dur() {
  digitalWrite(motorA1, LOW);
  digitalWrite(motorA2, LOW);
  digitalWrite(motorB1, LOW);
  digitalWrite(motorB2, LOW);
}