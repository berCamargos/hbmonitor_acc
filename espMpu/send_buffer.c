#include "send_buffer.h"
#include "uart.h"
#include "main.h"

#define TRANS_DEBUG         0
#define ESCAPE_CHAR         73
#define START_CHAR          74

extern uint8_t buffer[BUFFER_SIZE];

uint16_t input_pos, output_pos, last_pos;
uint8_t looped = 0;

void add_pkt(uint8_t *buff, uint8_t bufflen, uint8_t type, uint32_t timestamp)
{
    uint8_t i, revi;
    i = 0;
    while(i < bufflen)
    {
        if(buff[i] == START_CHAR || buff[i] == ESCAPE_CHAR)
        {
            /* First move the rest of the string forward */
            for(revi = bufflen; revi > i; revi--)
            {
                buff[revi] = buff[revi-1];
            }
            buff[i] = ESCAPE_CHAR;
            i += 1;
            bufflen += 1;
        }
        i += 1;
    }

    bufflen += 6;
    for(revi = bufflen; revi > 5; revi--)
    {
        buff[revi] = buff[revi-6];
    }

    buff[0] = START_CHAR;
    memcpy(&buff[2], &timestamp, 4);
    buff[1] = type;

    if ((input_pos + bufflen) >= BUFFER_SIZE)
    {
        last_pos = input_pos;
        input_pos = 0;
        looped += 1;
    }
    memcpy(&buffer[input_pos], buff, bufflen);
#if DEBUG
    if (type == 2)
    {
        os_printf("\nadd   : %d|%d|%d", buffer[input_pos + bufflen-2], buffer[input_pos + bufflen-1], bufflen);
    }
#endif
    input_pos += bufflen;
}

void send_timerfunc()
{
    uint16_t send_len = 0, next_output;
    if (looped)
    {
        if (output_pos < last_pos)
        {
            send_len = (last_pos - output_pos);
        } else {
            output_pos = 0;
        }
        next_output = 0;
        looped = 0;
    } else {
        if (output_pos < input_pos)
        {
            send_len = (input_pos - output_pos);
            next_output = output_pos + send_len;
        }
    }
    if (send_len > 0)
    {
#if DEBUG
        os_printf("\nsend   : %d|%d|%d", input_pos, output_pos, send_len);
#else

#if SEND_UART
        uart0_tx_buffer(&buffer[output_pos], send_len);
#else
        espconn_send(&server, buffer, temp_globaloffset + 8);
#endif
#endif
        output_pos = next_output;
    }
    os_timer_arm(&sendTimer, 10, 0);
}
