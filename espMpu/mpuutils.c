#include "mpuutils.h"
#include "main.h"
#include "send_buffer.h"


#define MAX_I2C_RETRIES     10
#define BUFFER_MPU_SIZE     512

#define BUFFER_ADC_SIZE     10

extern uint8_t abort_readings;

static uint8_t buffer_mpu[BUFFER_MPU_SIZE];
static uint32_t mpu_cntr = 0;
static uint8_t i2caddr = 0;



uint8_t ICACHE_FLASH_ATTR writeVal(uint8_t addr, uint8_t writeaddr, uint8_t val)
{

    uint8_t gotAck = 0, counter = 0;
    while ((!gotAck) && (counter++ < MAX_I2C_RETRIES))
    {
        i2c_master_start();
        i2c_master_writeByte(addr << 1 | 0x00);
        system_soft_wdt_feed();
        if(!i2c_master_checkAck())
        {
            wait100ns(10);
            i2c_master_stop();
            continue;
        }
        i2c_master_writeByte(writeaddr);
        gotAck = i2c_master_checkAck();
        i2c_master_writeByte(val);
        gotAck *= i2c_master_checkAck();
        i2c_master_stop();
    }
    return gotAck;
}

uint8_t ICACHE_FLASH_ATTR readVal(uint8_t addr, uint8_t writeaddr, uint8_t len, uint8_t *values)
{
    uint8_t gotAck = 0, i, counter = 0;
    while ((!gotAck) && (counter++ < MAX_I2C_RETRIES))
    {
        i2c_master_start();
        i2c_master_writeByte(addr << 1 | 0x00);
        if(!i2c_master_checkAck())
        {
            wait100ns(10);
            i2c_master_stop();
            continue;
        }
        i2c_master_writeByte(writeaddr);
        gotAck += i2c_master_checkAck();
        i2c_master_stop();
        i2c_master_start();
        i2c_master_writeByte(addr << 1 | 0x01);
        gotAck = i2c_master_checkAck();
        if(!gotAck)
        {
            wait100ns(10);
            i2c_master_stop();
            continue;
        }
        for(i = 0; i < len; i++)
        {
            values[i] = i2c_master_readByte();
            if (i != (len-1))
            {
                i2c_master_send_ack();
            }
        }
        /*Need to add a NACK*/
        i2c_master_send_nack();
        i2c_master_stop();
    }
    return gotAck;
}

void ICACHE_FLASH_ATTR initMPU(uint8_t addr)
{
    uint8_t gotAck = 0;
   
    //gotAck += writeVal(addr, 107, 128); /*Reset MPU*/
    writeVal(addr, 0x1A, (64*0+1)); /*Set FIFO_MODE FIFO_MODE = 0 -> overflow  | Set DLPF_CFG to 1->1Khz, it needs to be between 1 and 6 for the FIFO sample rate to work*/
    gotAck += writeVal(addr, 0x23, READ_ACC*8 + READ_GY*(16 + 32 + 64)); /*Set FIFO_EN to enable only ACC*/
    gotAck += writeVal(addr, 0x6A, (64+4+1)); /*Set USER_CTRL Enable FIFO, Reset Signals, reset CounterFIFO, Reset Signals, reset CounterFIFO, Reset Signals, reset Counter*/
    gotAck += writeVal(addr, 0x6B, 0); /*All is enable?*/
    gotAck += writeVal(addr, 0x37, 16); /*Pin will be up uintil interrupt status cleared*/
    gotAck += writeVal(addr, 0x38, 16); /*Interrupt Enable FIFO OVERFLOW*/
    gotAck += writeVal(addr, 28, 0); /*Scale acc to +-2g*/
    gotAck += writeVal(addr, 29, 1); /*Acc freq to 1kHz*/
    gotAck += writeVal(addr, 25, 4); /*Sample rate to 1kHz*/
    gotAck += writeVal(addr, 27, 0); /*Set Fchoice_b to 00 (Fchoice to 11) -> this allows for the FIFO sample rate to work*/

    uint8_t teste;
    readVal(addr, 0x75, 1, &teste); 
#if DEBUG
    os_printf("\n(%d)%d\n",gotAck, teste);
#endif
    wait100ns(100);

    writeVal(addr, 0x6A, 64+4); /*Reset FIFO*/
}

uint16_t getFifoSize(uint8_t addr)
{
    uint16_t fifosize = 0;
    uint8_t temp;
    readVal(addr, 0x72, 2, (uint8_t *)&fifosize);
    temp = ((uint8_t *)&fifosize)[0];
    ((uint8_t *)&fifosize)[0] = ((uint8_t *)&fifosize)[1];
    ((uint8_t *)&fifosize)[1] = temp;

    if (fifosize > 512)
    {
        fifosize = 0;
    }
    return fifosize;
}

uint8_t find_mpu_id()
{
    uint8_t gotAck = 0;
    while(1)
    {
        i2c_master_start();
        i2c_master_writeByte(i2caddr << 1 | 0x00);
        gotAck = i2c_master_checkAck();
        i2c_master_stop();
        if(gotAck)
        {
            initMPU(i2caddr);
#if DEBUG
            os_printf("\nFound MPU with ID: %d\n",(i2caddr));
#endif
            break;
        } else {
#if DEBUG
            os_printf("%d,",i2caddr);
#endif
            i2caddr += 1;
            if (i2caddr == 0)
            {
#if DEBUG
                os_printf("\nCouldn't find MPU will try again in 1s\n");
#endif
                return;
            }
        }
    }
}

void ICACHE_FLASH_ATTR i2c_timerfunc(void *arg)
{
    if ((abort_readings) || (i2caddr == 0))
    {
        abort_readings = 0;
        return;
    }
    uint16_t fifosize, readsize, counter, offset;
    //for(counter = 0; counter < 1000; counter++)
    fifosize = getFifoSize(i2caddr);
    if(fifosize > 36)
    {

        if (fifosize > 0xFF)
        {
            readsize = 0xFF;
            add_pkt(buffer_mpu, 0, PCK_MPU_FULL, system_get_time());
        } else {
            readsize = fifosize;
        }


        /*Need to make sure that reading is multiple of SIZE*/
#if DEBUG
        os_printf("\nmpu    : %d, %d, %d", readsize, fifosize, mpu_cntr);
#endif
        readsize = DATA_SIZE*((uint16_t)readsize/DATA_SIZE);
        readVal(i2caddr, 0x74, readsize, buffer_mpu);
        add_pkt(buffer_mpu, readsize, PCK_MPU_DATA, mpu_cntr);
#if DEBUG
        os_printf("\nmpu    : %d, %d, %d", readsize, fifosize, mpu_cntr);
#endif
        fifosize -= readsize;
        mpu_cntr += readsize;
        system_soft_wdt_feed();

    } 
    else if(fifosize == 0)
    {
        add_pkt(buffer_mpu, 0, PCK_MPU_TST, system_get_time());
        mpu_cntr = 0;
    }
    else 
    {
#if DEBUG
        os_printf("\nmpu_tst: %d", system_get_time());
#endif
        //mpu_cntr = 0;
        //add_pkt(buffer_mpu, 0, PCK_MPU_TST, system_get_time());
    }
    os_timer_arm(&i2cTimer, 10, 0);
}

volatile uint32_t laststart = 0, T = 0;
volatile uint16_t adc_period = ADC_PERIOD;
void ICACHE_FLASH_ATTR adc_timerfunc(void *arg)
{

    uint8_t buffer_adc[BUFFER_ADC_SIZE];
    T = (system_get_time() - laststart)/1000;
    if (laststart != 0)
    {
        if (T > ADC_PERIOD)
        {
            adc_period -= 1;
        } else {
            adc_period += 1;
        }
    } else {
        adc_period = ADC_PERIOD;
    }
    if (adc_period < 1)
    {
        adc_period = 1;
    }
    laststart = system_get_time();
#if ADC_PLOT 
    os_printf("------------\n%d, %d", T, adc_period);
#endif
    uint16_t adc_read = system_adc_read();
#if ADC_PLOT 
    os_printf("|%d|", adc_read);
#endif
    //adc_read /= 4;
#if ADC_PLOT 
    os_printf("|%d|", adc_read);
#endif

#if DEBUG
    os_printf("\nadc    : %d, %d", adc_read, system_get_time());
#endif
    memcpy(buffer_adc, &adc_read, 2);
    add_pkt(buffer_adc, 2, PCK_ADC_DATA, system_get_time());
    os_timer_arm(&adcTimer, adc_period, 0);
#if ADC_PLOT 
    os_printf("%d\n--------------", system_get_time() - laststart);
#endif
}
