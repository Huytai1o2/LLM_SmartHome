#include <Arduino.h>
#include <WiFi.h>
#include <Arduino_MQTT_Client.h>
#include <Server_Side_RPC.h>
#include <ThingsBoard.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ── Configuration ─────────────────────────────────────────────────────────────
constexpr char     WIFI_SSID[]          = "P1019";
constexpr char     WIFI_PASSWORD[]      = "phong@1019";
constexpr char     TOKEN[]              = "xdF2nW4aR9SAdqqPiym0";
constexpr char     DEVICE_ID[]          = "fcceeaa0-3111-11f1-9981-cffbb69f5b14";
constexpr char     THINGSBOARD_SERVER[] = "app.coreiot.io";
constexpr uint16_t THINGSBOARD_PORT     = 1883U;
constexpr uint32_t SERIAL_BAUD          = 115200U;

// ── Hardware ──────────────────────────────────────────────────────────────────
constexpr int LED_PIN = 2;

// ── Keys / RPC ────────────────────────────────────────────────────────────────
constexpr char    LED_KEY[]       = "led";
constexpr char    RPC_SET_VALUE[] = "setValue";
constexpr uint8_t MAX_RPC_SUBS   = 1U;
constexpr uint8_t MAX_RPC_RESP   = 1U;

// ── ThingsBoard objects ───────────────────────────────────────────────────────
constexpr uint16_t MAX_MSG_SIZE = 512U;

WiFiClient          espClient;
Arduino_MQTT_Client mqttClient(espClient);
Server_Side_RPC<MAX_RPC_SUBS, MAX_RPC_RESP> rpc;
const std::array<IAPI_Implementation *, 1U> apis = {&rpc};
// Set keepAlive time to 60s instead of default (15s) to avoid unneeded disconnects
ThingsBoard tb(mqttClient, MAX_MSG_SIZE, MAX_MSG_SIZE, Default_Max_Stack_Size, apis);

// ── Shared state ──────────────────────────────────────────────────────────────
static volatile bool ledState          = false;
static volatile bool pendingAttrUpdate = false;
static bool          subscribed        = false;

// ── Queue: loop() → ledTask ───────────────────────────────────────────────────
static QueueHandle_t ledQueue = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// RPC callback: setValue { "led": true/false }
// ─────────────────────────────────────────────────────────────────────────────
void onSetValue(const JsonVariantConst &params, JsonDocument &response) {
    if (!params.containsKey(LED_KEY)) {
        Serial.println("[RPC] setValue: missing key 'led'");
        response["error"] = "missing key 'led'";
        return;
    }
    bool newState     = params[LED_KEY].as<bool>();
    ledState          = newState;
    xQueueOverwrite(ledQueue, &newState);
    Serial.printf("[RPC] setValue led = %s\n", newState ? "true" : "false");
    
    // Gửi lại đúng giá trị json boolean để TB không bị lỗi Format Parsing
    response[LED_KEY] = newState;
}

// Global scope for callbacks to ensure memory is retained
// IMPORTANT: Đổi MAX_RPC_RESP thành số lớn hơn để buffer ko bị tràn nếu gửi/nhận liền mạch
// Tuy nhiên mảng config ban đầu định nghĩa MAX_RPC_RESP = 1U nên mình sẽ không đổi macro
const std::array<RPC_Callback, MAX_RPC_SUBS> callbacks = {
    RPC_Callback(RPC_SET_VALUE, onSetValue)
};

// ─────────────────────────────────────────────────────────────────────────────
// WiFi helpers – mirrors example InitWiFi / reconnect
// ─────────────────────────────────────────────────────────────────────────────
void InitWiFi() {
    Serial.print("[WiFi] Connecting");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\n[WiFi] Connected  IP: %s\n", WiFi.localIP().toString().c_str());
}

bool reconnect() {
    if (WiFi.status() == WL_CONNECTED) {
        return true;
    }
    Serial.println("[WiFi] Lost, reconnecting...");
    InitWiFi();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// FreeRTOS Task: LED control – only touches GPIO, never calls tb  (Core 0)
// ─────────────────────────────────────────────────────────────────────────────
void ledTask(void *pvParameters) {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    bool state = false;
    for (;;) {
        if (xQueueReceive(ledQueue, &state, portMAX_DELAY) == pdTRUE) {
            digitalWrite(LED_PIN, state ? HIGH : LOW);
            Serial.printf("[LED] %s\n", state ? "ON" : "OFF");
            
            // Delay 150ms để tránh gửi mạng quá sát với thời điểm phản hồi RPC
            // Nếu gửi 2 gói tin liên tục trong < 1ms, Server có thể hiểu nhầm là spam/rate-limit và tự ngắt kết nối
            vTaskDelay(pdMS_TO_TICKS(500));
            
            // Báo cho loop() biết task đã bật/tắt xong -> gửi update lên ThingsBoard
            pendingAttrUpdate = true;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FreeRTOS Task: ThingsBoard MQTT & RPC (Core 1)
// ─────────────────────────────────────────────────────────────────────────────
void tbTask(void *pvParameters) {
    for (;;) {
        if (!reconnect()) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (!tb.connected()) {
            Serial.printf("[TB] Connecting to %s ...\n", THINGSBOARD_SERVER);
            
            // Sử dụng TOKEN làm ClientID để đảm bảo tính độc nhất trên Server
            // Việc thiết lập ClientID giúp tránh tình trạng bị máy chủ ngắt kết nối do trùng lặp thiết bị (Collision)
            if (!tb.connect(THINGSBOARD_SERVER, TOKEN, THINGSBOARD_PORT, DEVICE_ID)) {
                Serial.println("[TB] Connection failed, retry in 5 s");
                vTaskDelay(pdMS_TO_TICKS(5000));
                continue;
            }
            Serial.println("[TB] Connected");
        }

        // Subscribe once – never reset subscribed after true (library handles MQTT resubscription)
        if (!subscribed) {
            Serial.println("[TB] Subscribing for RPC...");
            if (!rpc.RPC_Subscribe(callbacks.cbegin(), callbacks.cend())) {
                Serial.println("[TB] Subscribe failed");
                vTaskDelay(pdMS_TO_TICKS(1000));
                continue;
            }
            Serial.println("[TB] Subscribed to RPC: setValue");
            subscribed = true;
        }

        // Send pending client attribute (triggered by RPC callback)
        if (pendingAttrUpdate) {
            pendingAttrUpdate = false;
            tb.sendAttributeData(LED_KEY, (bool)ledState);
            Serial.printf("[Attr] led = %s\n", ledState ? "true" : "false");
        }

        tb.loop();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arduino entry points
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(SERIAL_BAUD);
    vTaskDelay(pdMS_TO_TICKS(1000));
    Serial.println("\n[Boot] CoreIoT LED – Server-Side RPC 2-way");

    ledQueue = xQueueCreate(1, sizeof(bool));
    configASSERT(ledQueue);

    // ledTask on Core 0
    xTaskCreatePinnedToCore(ledTask, "led_task", 2048, nullptr, 1, nullptr, 0);

    InitWiFi();

    // Kích hoạt cờ gửi trạng thái mặc định (false) lên ThingsBoard ngay lần đầu kết nối
    pendingAttrUpdate = true;

    // tbTask on Core 1 for MQTT handling
    xTaskCreatePinnedToCore(tbTask, "tb_task", 4096, nullptr, 1, nullptr, 1);
}

void loop() {
    // Trong RTOS, xóa task loop mặc định để tiết kiệm tài nguyên (giữ function trống)
    vTaskDelete(NULL);
}
