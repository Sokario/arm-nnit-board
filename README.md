# ARM-NNIT-Board
ARM Neural Netowk inference and training on Board

## Requirements
### Hardware requirements
The following boards are required:
- *List ARM-NNIT-Board hardware requirements here*

### Software requirements
ARM-NNIT-Board makes use of the following libraries (automatically
imported by `mbed deploy` or `mbed import`):
- *List ARM-NNIT-Board software requirements here*

## Usage
To clone **and** deploy the project in one command, use `mbed import` and skip to the
target enabling instructions:
```shell
mbed import https://gitlab.com/catie_6tron/arm-nnit-board.git arm-nnit-board
```

Alternatively:

- Clone to "arm-nnit-board" and enter it:
  ```shell
  git clone https://gitlab.com/catie_6tron/arm-nnit-board.git arm-nnit-board
  cd arm-nnit-board
  ```

- Deploy software requirements with:
  ```shell
  mbed deploy
  ```

Enable the custom target:
```shell
cp zest-core-stm32h753zi/custom_targets.json .
```

Compile the project:
```shell
mbed compile
```

Program the target device with a Segger J-Link debug probe and
[`sixtron_flash`](https://gitlab.com/catie_6tron/6tron-flash) tool:
```shell
sixtron_flash stm32h753zi BUILD/ZEST_CORE_STM32H753ZI/GCC_ARM/arm-nnit-board.elf
```

Debug on the target device with the probe and Segger
[Ozone](https://www.segger.com/products/development-tools/ozone-j-link-debugger)
software.
