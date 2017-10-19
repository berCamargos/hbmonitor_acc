#include "ets_sys.h"
#include "osapi.h"
#include "os_type.h"
#include "user_interface.h"
#include "c_types.h"
#include "espconn.h"
#include "main.h"

#include "mpuutils.h"
#include "send_buffer.h"

#include "espmissingincludes.h"

#define ADC_BUFFER_SIZE 128



#define MAGIC_NUMBER        123456
#define MAGIC_NUMBER_ADC    678901

#define USE_AP          1


#define AP_SSID         "acc_teste"
#define AP_PASSWD     "teste_acc"

/* Funcionou com o ESP8266EX_Demo board, recebeu ack dos ids 209 e 208 (read e write) 
 * Usa pullup de 2.2k documentação na pasta 
 *
 *
 *
 * */

#if USE_AP
static uint8_t ipaddr[4] = {192,168,43,1};
#else
static uint8_t ipaddr[4] = {192,168,1,247};
#endif
static uint16_t maxAdc,sumAdc,numSamples;
static adc_data_t adcData;
static struct espconn server;
static esp_tcp tcpServer;
uint8_t temp_buffer[BUFFER_SIZE];
uint8_t adc_buffer[ADC_BUFFER_SIZE];
uint8_t sentfile = 0;
uint16_t adc_globaloffset = 8;

uint8_t abort_readings = 0;
uint16_t globaloffset = 8;
uint8_t buffer[BUFFER_SIZE];

extern os_timer_t i2cTimer, adcTimer, sendTimer, connectTimer, sendTimeout;

static void connect_tcp();
static void disconnect_tcp();




void ICACHE_FLASH_ATTR connect_timerfunc(void *arg)
{
#if DEBUG
    os_printf("Trying to connect to TCP\n");
#endif
    tcpServer.remote_port = SERVER_PORT;
    memcpy(tcpServer.remote_ip, ipaddr, 4);
    
     //set up the local IP
    struct ip_info ipconfig;
    wifi_get_ip_info(STATION_IF, &ipconfig);
    os_memcpy(tcpServer.local_ip, &ipconfig.ip, 4);
    
    server.type = ESPCONN_TCP;
    server.state = ESPCONN_NONE;
    server.proto.tcp = &tcpServer;
    espconn_regist_connectcb(&server, connect_tcp);
    espconn_regist_disconcb(&server, disconnect_tcp);
    if(espconn_connect(&server) == 0)
    {
        os_timer_arm(&connectTimer, 1000, 0);
#if DEBUG
        os_printf("Failed to Connect\n");
#endif
    }
}

void ICACHE_FLASH_ATTR sendTimeout_timerfunc(void *arg)
{
 #if DEBUG
    os_printf("Send TCP timeout, disconnecting\n");
#endif   
    espconn_disconnect(&server);
    disconnect_tcp();
}



static inline unsigned get_ccount(void)
{
    unsigned r;
    asm volatile ("rsr %0, ccount" : "=r"(r));
    return r;
}

static void ICACHE_FLASH_ATTR startAll() 
{
    i2c_master_gpio_init();
    os_timer_arm(&i2cTimer, 1000, 0);
    os_timer_arm(&adcTimer, ADC_PERIOD, 0);
#if SEND_UART
    os_timer_arm(&sendTimer, SEND_PERIOD, 0);
#endif
    find_mpu_id();
}

static void ICACHE_FLASH_ATTR stopAll()
{
    os_timer_disarm(&i2cTimer);
    os_timer_disarm(&adcTimer);
}

static void connect_tcp()
{
#if DEBUG
    os_printf("TCP Connected\n");
#endif
    os_timer_disarm(&connectTimer);
    
    espconn_regist_sentcb(&server, send_timerfunc);
    espconn_send(&server,(uint8_t *)&adcData,sizeof(adc_data_t));
    startAll();
}

static void disconnect_tcp()
{
#if DEBUG
    os_printf("TCP Disconnected\n");
#endif
    os_timer_disarm(&sendTimeout);
    os_timer_arm(&connectTimer, 1000, 0);
    abort_readings = 1;
}

void ICACHE_FLASH_ATTR wifi_sta_callback( System_Event_t *evt )
{
    switch ( evt->event )
    {
        case EVENT_STAMODE_CONNECTED:
        {
#if DEBUG
            os_printf("Connected\n");
#endif
            break;
        }

        case EVENT_STAMODE_DISCONNECTED:
        {
#if DEBUG
            os_printf("Disconnected\n");
#endif
            break;
        }

        case EVENT_STAMODE_GOT_IP:
        {
#if DEBUG
            os_printf("Got IP\n");
#endif
            os_timer_arm(&connectTimer, 100, 0);
            break;
        }
    
        case EVENT_SOFTAPMODE_STACONNECTED:
        {
#if DEBUG
            os_printf("CONNECTED!!!!!:)");
#endif
        }

        default:
        {
            break;
        }
    }
}

void connectToWifi()
{
    wifi_set_event_handler_cb( wifi_sta_callback );
#if USE_AP
    wifi_set_opmode( STATION_MODE );
    wifi_promiscuous_enable(0);
    
    struct station_config config;
    
    config.bssid_set = 0;
    os_memcpy( &config.ssid, AP_SSID, 32 );
    os_memcpy( &config.password, AP_PASSWD, 64 );
    config.bssid_set = 0;
    wifi_station_set_config( &config );
#else
    wifi_set_opmode( STATION_MODE );
    
    struct station_config config;
    
    config.bssid_set = 0;
    os_memcpy( &config.ssid, WIFI_SSID, 32 );
    os_memcpy( &config.password, WIFI_PASSWD, 64 );
    wifi_station_set_config( &config );
#endif                    
#if ADC_PLOT 
    os_timer_arm(&adcTimer, ADC_PERIOD, 0);
#endif
}

void user_init(void)
{
    system_restore();
    gpio_init();
    struct rst_info* info = system_get_rst_info();
    uart_div_modify(0, UART_CLK_FREQ / 115200); 
#if DEBUG
	os_printf("\n\n\n\n\n\n\n\n\n\n\n\r------------------\r\nThis is Bernardo's masters code, for MPU and ADC reading\n\rReset Reason:%d\n\rBoot Mode:%d\n\r---------------------\n\r",info->reason,system_get_boot_mode());
#endif
    os_timer_setfn(&adcTimer, adc_timerfunc, NULL);
    os_timer_setfn(&i2cTimer, i2c_timerfunc, NULL);
    os_timer_setfn(&connectTimer, connect_timerfunc, NULL);
    os_timer_setfn(&sendTimeout, sendTimeout_timerfunc, NULL);
    os_timer_setfn(&sendTimer, send_timerfunc, NULL);
    
#if SEND_UART
    system_init_done_cb(startAll);
#else
    system_init_done_cb(connectToWifi);
#endif
}
