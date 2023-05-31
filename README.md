# MobileNet on ARM cortex-M7 architecture

This project, as the title says, enable the quantized Google Neural Network called MobileNet
on an ARM cortex-M7 architecture embedded on the STM32 NUCLEO-H743ZI.

The repo contains 2 directories
1.int8iq
2.q7

int8iq\MobileNet: contains the project with the whole MobilNet
int8iq\Single Layer: contains the projects of with the singol layers Depthwise & Pointwise

-\src: contains the main.c;
-\CMSIS\core: contains the header files of the adopted nn functions;
-\inc: contains the header files of the #define and input, parameters ans weights values;
-\NN_Lib\ConvolutionFunctions: adapted nn convolution functions;
-\NN_Lib\NNSupportFunctions: adapted nn support functions.




