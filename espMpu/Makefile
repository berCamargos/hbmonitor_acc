#
# Makefile for rBoot sample project
# https://github.com/raburton/esp8266
#

# use wifi settings from environment or hard code them here
WIFI_SSID ?= ""
WIFI_PWD  ?= ""

SDK_BASE   ?= /media/OldHome/bermcs/DEV/DEVGit/esp-open-sdk/sdk
#SDK_BASE   ?= /home/bercamargos/tools/espOpenSDK/esp-open-sdk/sdk
SDK_LIBDIR  = lib
SDK_INCDIR  = include

R_BOOT ?= $(SDK_BASE)/../../rboot

GCC_FOLDER:=$(SDK_BASE)/../xtensa-lx106-elf
XTENSA_BINDIR:=$(GCC_FOLDER)/bin

ESPTOOL2     ?= $(SDK_BASE)/../../esptool2/esptool2
ESPTOOL     ?= $(SDK_BASE)/../esptool/esptool.py
FW_SECTS      = .text .data .rodata
FW_USER_ARGS  = -quiet -bin -boot2

ifndef XTENSA_BINDIR
CC := xtensa-lx106-elf-gcc
LD := xtensa-lx106-elf-gcc
else
CC := $(addprefix $(XTENSA_BINDIR)/,xtensa-lx106-elf-gcc)
LD := $(addprefix $(XTENSA_BINDIR)/,xtensa-lx106-elf-gcc)
endif

BUILD_DIR = build
FIRMW_DIR = firmware

SDK_LIBDIR := $(addprefix $(SDK_BASE)/,$(SDK_LIBDIR))
SDK_INCDIR := $(addprefix -I$(SDK_BASE)/,$(SDK_INCDIR))

LIBS    = c gcc hal phy net80211 lwip wpa main2 pp crypto
CFLAGS  = -Os -g -O2 -Wpointer-arith -Wundef -Werror -Wno-implicit -Wl,-EL -fno-inline-functions -nostdlib -mlongcalls  -mtext-section-literals  -D__ets__ -DICACHE_FLASH
LDFLAGS = -nostdlib -u call_user_start -Wl,-static -u Cache_Read_Enable_New 

SRC		:= $(wildcard *.c)
OBJ		:= $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC))
LIBS		:= $(addprefix -l,$(LIBS))

ifneq ($(WIFI_SSID), "")
	CFLAGS += -DWIFI_SSID=\"$(WIFI_SSID)\"
endif
ifneq ($(WIFI_PWD), "")
	CFLAGS += -DWIFI_PWD=\"$(WIFI_PWD)\"
endif

.SECONDARY:
.PHONY: all clean

C_FILES = $(wildcard *.c)
O_FILES = $(patsubst %.c,$(BUILD_DIR)/%.o,$(C_FILES))

all: $(BUILD_DIR) $(FIRMW_DIR) $(FIRMW_DIR)/rom0.bin

aaa: $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c %.h
	@echo "CC $<"
	@$(CC) -I. $(SDK_INCDIR) $(CFLAGS) -o $@ -c $<

$(BUILD_DIR)/%.elf: $(O_FILES)
	@echo "LD $(notdir $@)"
	@$(LD) -L$(SDK_LIBDIR) -T$(notdir $(basename $@)).ld $(LDFLAGS) -Wl,--start-group $(LIBS) $^ -Wl,--end-group -o $@

$(FIRMW_DIR)/%.bin: $(BUILD_DIR)/%.elf
	@echo "FW $(notdir $@)"
	@$(ESPTOOL2) $(FW_USER_ARGS) $^ $@ $(FW_SECTS)

$(BUILD_DIR):
	@mkdir -p $@

$(FIRMW_DIR):
	@mkdir -p $@

burn1:
	python2 $(ESPTOOL) -b 230400 --port /dev/ttyUSB1 write_flash -fs 32m 0x00000 $(R_BOOT)/$(FIRMW_DIR)/rboot.bin 0x02000 $(FIRMW_DIR)/rom0.bin
burn0:
	python2 $(ESPTOOL) -b 230400 --port /dev/ttyUSB0 write_flash -fs 32m 0x00000 $(R_BOOT)/$(FIRMW_DIR)/rboot.bin 0x02000 $(FIRMW_DIR)/rom0.bin

clean:
	@echo "RM $(BUILD_DIR) $(FIRMW_DIR)"
	@rm -rf $(BUILD_DIR)
	@rm $(FIRMW_DIR)/rom0.bin
