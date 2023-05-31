/* ----------------------------------------------------------------------
* Copyright (C) 2010-2018 Arm Limited. All rights reserved.
*
*
* Project:       CMSIS NN Library
* Title:         arm_nnexamples_cifar10.cpp
*
* Description:   Convolutional Neural Network Example
*
* Target Processor: Cortex-M4/Cortex-M7
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the
*     distribution.
*   - Neither the name of Arm LIMITED nor the names of its contributors
*     may be used to endorse or promote products derived from this
*     software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
* -------------------------------------------------------------------- */

/**
 * @ingroup groupExamples
 */

/**
 * @defgroup CNNExample Convolutional Neural Network Example
 *
 * \par Description:
 * \par
 * Demonstrates a convolutional neural network (CNN) example with the use of convolution,
 * ReLU activation, pooling and fully-connected functions.
 *
 * \par Model definition:
 * \par
 * The CNN used in this example is based on CIFAR-10 example from Caffe [1].
 * The neural network consists
 * of 3 convolution layers interspersed by ReLU activation and max pooling layers, followed by a
 * fully-connected layer at the end. The input to the network is a 32x32 pixel color image, which will
 * be classified into one of the 10 output classes.
 * This example model implementation needs 32.3 KB to store weights, 40 KB for activations and
 * 3.1 KB for storing the \c im2col data.
 *
 * \image html CIFAR10_CNN.gif "Neural Network model definition"
 *
 * \par Variables Description:
 * \par
 * \li \c conv1_wt is convolution layer weight matrices
 * \li \c conv1_bias, \c conv2_bias, \c conv3_bias are convolution layer bias arrays
 * \li \c ip1_wt, ip1_bias point to fully-connected layer weights and biases
 * \li \c input_data points to the input image data
 * \li \c output_data points to the classification output
 * \li \c col_buffer is a buffer to store the \c im2col output
 * \li \c scratch_buffer is used to store the activation data (intermediate layer outputs)
 *
 * \par CMSIS DSP Software Library Functions Used:
 * \par
 * - arm_convolve_HWC_q7_fast()
 *
 */

#include <stdint.h>
#include <stdio.h>
#include "arm_math.h"
#include "stm32h7xx.h"
#include "stm32h7xx_nucleo_144.h"

#include <arm_nnexamples_Point_parameter.h>
#include <arm_nnexamples_Point_weights.h>

#include "arm_nnfunctions.h"
#include <arm_nnexamples_Point_inputs.h>

// include the input and weights

static uint8_t conv1_wt[CONV1_IM_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH] = CONV1_WT; // MODIFICA q7_t --> uint8_t
static int16_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS; // MODIFICA q7_t --> int32_t
static q15_t M_ZERO = 24068;

/* Here the image_data should be the raw uint8 type RGB image in [RGB, RGB, RGB ... RGB] format */
uint8_t   image_data[CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM] = IMG_DATA; // MODIFICA q7_t --> uint8_t
uint8_t   output_data_1[CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH]; // MODIFICA q7_t --> uint8_t

//Per Verifica
uint8_t   output_data_verifica[CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH]=OUTPUT_DATA;

//vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
/*
 *col_buffer:
 * - stores the im2col(image to column) output for convolutional layers
 * - vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
 * - To find out the required col_buffer size across all convolutional layers, this formula below is applied.
 * 	 2*2*(conv # of filters)*(kernel width)*(kernel height)
 * 	 :#2 per convertirlo in 16 byte
 * 	 :#2 per lavorare con 2 layer contemporaneamente
 */
q7_t      col_buffer[2 * 2 * CONV1_IM_CH * CONV1_KER_DIM * CONV1_KER_DIM];

/*
 *scratch_buffer:
 * - stores the activation data (intermediate layer outputs)
 * - For the scratch buffer, which splits into two parts, for a given layer, one could serve as input while the other as output.
 *   Similarly, its maximum size can be determined by iterating over all layers.
 */
uint8_t      scratch_buffer[CONV1_IM_DIM * CONV1_IM_DIM * (128+256)];

/******** Cycle counter defines  **********/
volatile unsigned int cyc[2];
volatile unsigned int *DWT_CYCCNT = (volatile unsigned int *)0xE0001004; 	// Cycle counter
volatile unsigned int *DWT_CONTROL= (volatile unsigned int *)0xE0001000;	// counter control
volatile unsigned int *SCB_DEMCR  = (volatile unsigned int *)0xE000EDFC;
#define STOPWATCH_START {cyc[0]=*DWT_CYCCNT;} 								// start counting
#define STOPWATCH_STOP_1  {cyc[1]=*DWT_CYCCNT; cyc[1]=cyc[1]-cyc[0];}

int main()
{
	//Enable I-Cache
	SCB_EnableICache();
	//Enable D-Cache
	SCB_EnableDCache();

  /* start the execution */

  uint8_t     *img_buffer1 = scratch_buffer; // MODIFICA q7_t --> uint8_t
  uint8_t     *img_buffer2 = img_buffer1 + 5 * 5 * 128; // MODIFICA q7_t --> uint8_t

  /* input pre-processing */
  /* NEW */
  for (int i=0;i<5*5*128; i+=1) {
        img_buffer2[i] =   (uint8_t)image_data[i];
      }
  // init, reset and start the cycle counter
  		*SCB_DEMCR = *SCB_DEMCR | 0x01000000;
  		*DWT_CYCCNT = 0; 							// reset the counter
  		*DWT_CONTROL = *DWT_CONTROL | 1 ; 			// enable the counter


  		//STOPWATCH_START	/* Start counting cycles */
  		cyc[0]=*(volatile unsigned int *)0xE0001004;
 arm_convolve_HWC_int8iq_fast(img_buffer2, CONV1_IM_DIM, CONV1_IM_CH,
							  conv1_wt, CONV1_ZA, CONV1_ZB, CONV1_Z, M_ZERO, N,
							  CONV1_OUT_CH, CONV1_KER_DIM,
							  CONV1_PADDING, CONV1_STRIDE, conv1_bias,
							  img_buffer1, CONV1_OUT_DIM, (q15_t *) col_buffer, NULL);
  		//STOPWATCH_STOP_1 /* Stop counting cycles, result is in cyc[1] */
 	 	 cyc[1]=*(volatile unsigned int *)0xE0001004;
 	 	 cyc[1]=cyc[1]-cyc[0];

  return 0;
}
