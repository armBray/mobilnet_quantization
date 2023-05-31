/* ----------------------------------------------------------------------
* Copyright (C) 2010-2018 Arm Limited. All rights reserved.
*
*
* Project:       ND
* Title:         ND
*
*
* Target Processor: Cortex-M4/Cortex-M7
*
*/
#include <160-0.25-Mobile_Net_inputs.h>
#include <160-0.25-Mobile_Net_parameter.h>
#include <160-0.25-Mobile_Net_weights.h>
#include "stm32h7xx.h"
#include "stm32h7xx_nucleo_144.h"

#include <stdint.h>
#include <stdio.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"

//weights
/*
 * Per la Convolutional il kernel è così definito
 * Dk*Dk*N*M dove M sono i canali di INPUT e N sono i canali di OUTPUT
 *
 * Per la depthwise il kernel è così definito
 * Dk*Dk*1*M dove M sono i canali di INPUT
 *
 * mentre per la pointwise
 * 1*1*M*N dove N sono i canali i OUTPUT
 *
 */

//Layer 1	Conv / s2
const q7_t conv1_wt[CONV1_KER_DIM * CONV1_KER_DIM * CONV1_IM_CH * CONV1_OUT_CH] = CONV1_WT;
const q7_t conv1_bias[CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH] = CONV1_BIAS;

//Layer 2	Conv dw/ s1
const q7_t conv2_wt[CONV2_KER_DIM * CONV2_KER_DIM * 1 * CONV2_OUT_CH] = CONV2_WT;
const q7_t conv2_bias[CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH] = CONV2_BIAS;
//Layer 3	Conv Point/ s1
const q7_t conv3_wt[CONV3_KER_DIM * CONV3_KER_DIM * CONV3_IM_CH * CONV3_OUT_CH] = CONV3_WT;
const q7_t conv3_bias[CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH] = CONV3_BIAS;

//Layer 4	Conv dw/ s2
const q7_t conv4_wt[CONV4_KER_DIM * CONV4_KER_DIM * 1 * CONV4_OUT_CH] = CONV4_WT;
const q7_t conv4_bias[CONV4_OUT_DIM * CONV4_OUT_DIM * CONV4_OUT_CH] = CONV4_BIAS;
//Layer 5	Conv Point/ s1
const q7_t conv5_wt[CONV5_KER_DIM * CONV5_KER_DIM * CONV5_IM_CH * CONV5_OUT_CH] = CONV5_WT;
const q7_t conv5_bias[CONV5_OUT_DIM * CONV5_OUT_DIM * CONV5_OUT_CH] = CONV5_BIAS;

//Layer 6	Conv dw/ s1
const q7_t conv6_wt[CONV6_KER_DIM * CONV6_KER_DIM * 1 * CONV6_OUT_CH] = CONV6_WT;
const q7_t conv6_bias[CONV6_OUT_DIM * CONV6_OUT_DIM * CONV6_OUT_CH] = CONV6_BIAS;
//Layer 7	Conv Point/ s1
const q7_t conv7_wt[CONV7_KER_DIM * CONV7_KER_DIM * CONV7_IM_CH * CONV7_OUT_CH] = CONV7_WT;
const q7_t conv7_bias[CONV7_OUT_DIM * CONV7_OUT_DIM * CONV7_OUT_CH] = CONV7_BIAS;

//Layer 8	Conv dw/ s2
const q7_t conv8_wt[CONV8_KER_DIM * CONV8_KER_DIM * 1 * CONV8_OUT_CH] = CONV8_WT;
const q7_t conv8_bias[CONV8_OUT_DIM * CONV8_OUT_DIM * CONV8_OUT_CH] = CONV8_BIAS;
//Layer 9	Conv Point/ s1
const q7_t conv9_wt[CONV9_KER_DIM * CONV9_KER_DIM * CONV9_IM_CH * CONV9_OUT_CH] = CONV9_WT;
const q7_t conv9_bias[CONV9_OUT_DIM * CONV9_OUT_DIM * CONV9_OUT_CH] = CONV9_BIAS;

//Layer 10	Conv dw/ s1
const q7_t conv10_wt[CONV10_KER_DIM * CONV10_KER_DIM * 1 * CONV10_OUT_CH] = CONV10_WT;
const q7_t conv10_bias[CONV10_OUT_DIM * CONV10_OUT_DIM * CONV10_OUT_CH] = CONV10_BIAS;
//Layer 11	Conv Point/ s1
const q7_t conv11_wt[CONV11_KER_DIM * CONV11_KER_DIM * CONV11_IM_CH * CONV11_OUT_CH] = CONV11_WT;
const q7_t conv11_bias[CONV11_OUT_DIM * CONV11_OUT_DIM * CONV11_OUT_CH] = CONV11_BIAS;

//Layer 12	Conv dw/ s2
const q7_t conv12_wt[CONV12_KER_DIM * CONV12_KER_DIM * 1 * CONV12_OUT_CH] = CONV12_WT;
const q7_t conv12_bias[CONV12_OUT_DIM * CONV12_OUT_DIM * CONV12_OUT_CH] = CONV12_BIAS;
//Layer 13	Conv Point/ s1
const q7_t conv13_wt[CONV13_KER_DIM * CONV13_KER_DIM * CONV13_IM_CH * CONV13_OUT_CH] = CONV13_WT;
const q7_t conv13_bias[CONV13_OUT_DIM * CONV13_OUT_DIM * CONV13_OUT_CH] = CONV13_BIAS;

//Layer 14	Conv dw/ s1
const q7_t conv14_wt[CONV14_KER_DIM * CONV14_KER_DIM * 1 * CONV14_OUT_CH] = CONV14_WT;
const q7_t conv14_bias[CONV14_OUT_DIM * CONV14_OUT_DIM * CONV14_OUT_CH] = CONV14_BIAS;
//Layer 15	Conv Point/ s1
const q7_t conv15_wt[CONV15_KER_DIM * CONV15_KER_DIM * CONV15_IM_CH * CONV15_OUT_CH] = CONV15_WT;
const q7_t conv15_bias[CONV15_OUT_DIM * CONV15_OUT_DIM * CONV15_OUT_CH] = CONV15_BIAS;

//Layer 16	Conv dw/ s1
const q7_t conv16_wt[CONV16_KER_DIM * CONV16_KER_DIM * 1 * CONV16_OUT_CH] = CONV16_WT;
const q7_t conv16_bias[CONV16_OUT_DIM * CONV16_OUT_DIM * CONV16_OUT_CH] = CONV16_BIAS;
//Layer 17	Conv Point/ s1
const q7_t conv17_wt[CONV17_KER_DIM * CONV17_KER_DIM * CONV17_IM_CH * CONV17_OUT_CH] = CONV17_WT;
const q7_t conv17_bias[CONV17_OUT_DIM * CONV17_OUT_DIM * CONV17_OUT_CH] = CONV17_BIAS;

//Layer 18	Conv dw/ s1
const q7_t conv18_wt[CONV18_KER_DIM * CONV18_KER_DIM * 1 * CONV18_OUT_CH] = CONV18_WT;
const q7_t conv18_bias[CONV18_OUT_DIM * CONV18_OUT_DIM * CONV18_OUT_CH] = CONV18_BIAS;
//Layer 19	Conv Point/ s1
const q7_t conv19_wt[CONV19_KER_DIM * CONV19_KER_DIM * CONV19_IM_CH * CONV19_OUT_CH] = CONV19_WT;
const q7_t conv19_bias[CONV19_OUT_DIM * CONV19_OUT_DIM * CONV19_OUT_CH] = CONV19_BIAS;

//Layer 20	Conv dw/ s1
const q7_t conv20_wt[CONV20_KER_DIM * CONV20_KER_DIM * 1 * CONV20_OUT_CH] = CONV20_WT;
const q7_t conv20_bias[CONV20_OUT_DIM * CONV20_OUT_DIM * CONV20_OUT_CH] = CONV20_BIAS;
//Layer 21	Conv Point/ s1
const q7_t conv21_wt[CONV21_KER_DIM * CONV21_KER_DIM * CONV21_IM_CH * CONV21_OUT_CH] = CONV21_WT;
const q7_t conv21_bias[CONV21_OUT_DIM * CONV21_OUT_DIM * CONV21_OUT_CH] = CONV21_BIAS;

//Layer 22	Conv dw/ s1
const q7_t conv22_wt[CONV22_KER_DIM * CONV22_KER_DIM * 1 * CONV22_OUT_CH] = CONV22_WT;
const q7_t conv22_bias[CONV22_OUT_DIM * CONV22_OUT_DIM * CONV22_OUT_CH] = CONV22_BIAS;
//Layer 23	Conv Point/ s1
const q7_t conv23_wt[CONV23_KER_DIM * CONV23_KER_DIM * CONV23_IM_CH * CONV23_OUT_CH] = CONV23_WT;
const q7_t conv23_bias[CONV23_OUT_DIM * CONV23_OUT_DIM * CONV23_OUT_CH] = CONV23_BIAS;

//Layer 24	Conv dw/ s2
const q7_t conv24_wt[CONV24_KER_DIM * CONV24_KER_DIM * 1 * CONV24_OUT_CH] = CONV24_WT;
const q7_t conv24_bias[CONV24_OUT_DIM * CONV24_OUT_DIM * CONV24_OUT_CH] = CONV24_BIAS;
//Layer 25	Conv Point/ s1
const q7_t conv25_wt[CONV25_KER_DIM * CONV25_KER_DIM * CONV25_IM_CH * CONV25_OUT_CH] = CONV25_WT;
const q7_t conv25_bias[CONV25_OUT_DIM * CONV25_OUT_DIM * CONV25_OUT_CH] = CONV25_BIAS;

//Layer 26	Conv dw/ s1
const q7_t conv26_wt[CONV26_KER_DIM * CONV26_KER_DIM * 1 * CONV26_OUT_CH] = CONV26_WT;
const q7_t conv26_bias[CONV26_OUT_DIM * CONV26_OUT_DIM * CONV26_OUT_CH] = CONV26_BIAS;
//Layer 27	Conv Point/ s1
const q7_t conv27_wt[CONV27_KER_DIM * CONV27_KER_DIM * CONV27_IM_CH * CONV27_OUT_CH] = CONV27_WT;
const q7_t conv27_bias[CONV27_OUT_DIM * CONV27_OUT_DIM * CONV27_OUT_CH] = CONV27_BIAS;

//Layer 28	Avg Pool /s1
//Layer 29	FC /s1
const q7_t ip1_wt[IP1_DIM * IP1_OUT] = IP1_WT;
const q7_t ip1_bias[IP1_OUT] = IP1_BIAS;
//Layer 30	Softmax / s1


// Include the input
/* Here the image_data should be the raw uint8 type HxWxC image in [C,C,C,...,W,W,W,...H,H,H] format */
uint8_t   image_data[CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM] = IMG_DATA;
q7_t      output_data[IP1_OUT];

/*
 *col_buffer:
 * - stores the im2col(image to column) output for convolutional layers
 * - vector buffer: max(im2col buffer,average pool buffer, fully connected buffer)
 * - To find out the required col_buffer size across all convolutional layers, this formula below is applied.
 * 	 2*2*(conv # of filters)*(kernel width)*(kernel height)
 * 	 #2 16 byte
 * 	 #2 2 contemporaneamente
 */
q7_t      col_buffer[2 * 2 * 256 * 256]; //N.B.: gli ultimi 4 layer non vengono considerati!!!

/*
 *scratch_buffer:
 * - stores the activation data (intermediate layer outputs)
 * - For the scratch buffer, which splits into two parts, for a given layer, one could serve as input while the other as output.
 *   Similarly, its maximum size can be determined by iterating over all layers.
 */
q7_t      scratch_buffer[80 * 80 * 24];	//80*80*(8+16)

/******** Cycle counter defines  **********/
volatile unsigned int cyc[28];
volatile unsigned int *DWT_CYCCNT = (volatile unsigned int *)0xE0001004; 	// Cycle counter
volatile unsigned int *DWT_CONTROL= (volatile unsigned int *)0xE0001000;	// counter control
volatile unsigned int *SCB_DEMCR  = (volatile unsigned int *)0xE000EDFC;
//#define STOPWATCH_START {cyc[0]=*DWT_CYCCNT;} 								// start counting
//#define STOPWATCH_STOP_1  {cyc[1]=*DWT_CYCCNT; cyc[1]=cyc[1]-cyc[0];}			// stop counting, result is in cyc[1]
//#define STOPWATCH_STOP_2  {cyc[2]=*DWT_CYCCNT; cyc[2]=cyc[2]-cyc[0];}
//#define STOPWATCH_STOP_3  {cyc[3]=*DWT_CYCCNT; cyc[3]=cyc[3]-cyc[0];}
//#define STOPWATCH_STOP_4  {cyc[4]=*DWT_CYCCNT; cyc[4]=cyc[4]-cyc[0];}
//#define STOPWATCH_STOP_5  {cyc[5]=*DWT_CYCCNT; cyc[5]=cyc[5]-cyc[0];}
//#define STOPWATCH_STOP_6  {cyc[6]=*DWT_CYCCNT; cyc[6]=cyc[6]-cyc[0];}
//#define STOPWATCH_STOP_7  {cyc[7]=*DWT_CYCCNT; cyc[7]=cyc[7]-cyc[0];}
//#define STOPWATCH_STOP_8  {cyc[8]=*DWT_CYCCNT; cyc[8]=cyc[8]-cyc[0];}
//#define STOPWATCH_STOP_9  {cyc[9]=*DWT_CYCCNT; cyc[9]=cyc[9]-cyc[0];}
//#define STOPWATCH_STOP_10  {cyc[10]=*DWT_CYCCNT; cyc[10]=cyc[10]-cyc[0];}
//#define STOPWATCH_STOP_11  {cyc[11]=*DWT_CYCCNT; cyc[11]=cyc[11]-cyc[0];}
//#define STOPWATCH_STOP_12  {cyc[12]=*DWT_CYCCNT; cyc[12]=cyc[12]-cyc[0];}
//#define STOPWATCH_STOP_13  {cyc[13]=*DWT_CYCCNT; cyc[13]=cyc[13]-cyc[0];}
//#define STOPWATCH_STOP_14  {cyc[14]=*DWT_CYCCNT; cyc[14]=cyc[14]-cyc[0];}
//#define STOPWATCH_STOP_15  {cyc[15]=*DWT_CYCCNT; cyc[15]=cyc[15]-cyc[0];}
//#define STOPWATCH_STOP_16  {cyc[16]=*DWT_CYCCNT; cyc[16]=cyc[16]-cyc[0];}
//#define STOPWATCH_STOP_17  {cyc[17]=*DWT_CYCCNT; cyc[17]=cyc[17]-cyc[0];}
//#define STOPWATCH_STOP_18  {cyc[18]=*DWT_CYCCNT; cyc[18]=cyc[18]-cyc[0];}
//#define STOPWATCH_STOP_19  {cyc[19]=*DWT_CYCCNT; cyc[19]=cyc[19]-cyc[0];}
//#define STOPWATCH_STOP_20  {cyc[20]=*DWT_CYCCNT; cyc[20]=cyc[20]-cyc[0];}
//#define STOPWATCH_STOP_21  {cyc[21]=*DWT_CYCCNT; cyc[21]=cyc[21]-cyc[0];}
//#define STOPWATCH_STOP_22  {cyc[22]=*DWT_CYCCNT; cyc[22]=cyc[22]-cyc[0];}
//#define STOPWATCH_STOP_23  {cyc[23]=*DWT_CYCCNT; cyc[23]=cyc[23]-cyc[0];}
//#define STOPWATCH_STOP_24  {cyc[24]=*DWT_CYCCNT; cyc[24]=cyc[24]-cyc[0];}
//#define STOPWATCH_STOP_25  {cyc[25]=*DWT_CYCCNT; cyc[25]=cyc[25]-cyc[0];}
//#define STOPWATCH_STOP_26  {cyc[26]=*DWT_CYCCNT; cyc[26]=cyc[26]-cyc[0];}
//#define STOPWATCH_STOP_27  {cyc[27]=*DWT_CYCCNT; cyc[27]=cyc[27]-cyc[0];}

int main()
{
	//Enable I-Cache
	SCB_EnableICache();
	//Enable D-Cache
	SCB_EnableDCache();

	/* start the execution */
	  q7_t     *img_buffer1 = scratch_buffer;
	  q7_t     *img_buffer2 = img_buffer1 +  CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM;

	  /* input pre-processing */
	    for (int i=0;i < CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM; i+=1) {
	      img_buffer2[i] =   (q7_t)image_data[i];
	    }

	// init, reset and start the cycle counter
		*SCB_DEMCR = *SCB_DEMCR | 0x01000000;
		*DWT_CYCCNT = 0; 							// reset the counter
		*DWT_CONTROL = *DWT_CONTROL | 1 ; 			// enable the counter


		//STOPWATCH_START	/* Start counting cycles */
		cyc[0]=*(volatile unsigned int *)0xE0001004;
	/*Layer 1	Conv / s2
	 *Convolutional
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM,
								CONV1_PADDING, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT,
								img_buffer1, CONV1_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_1 /* Stop counting cycles, result is in cyc[1] */
		cyc[1]=*(volatile unsigned int *)0xE0001004;
		cyc[1]=cyc[1]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 2	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM,
	        		  	  	  	  	  			CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT,
	        									img_buffer2, CONV2_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_2
		cyc[2]=*(volatile unsigned int *)0xE0001004;
		cyc[2]=cyc[2]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 3	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM,
								CONV3_PADDING, CONV3_STRIDE, conv3_bias, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT,
								img_buffer1, CONV3_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_3
		cyc[3]=*(volatile unsigned int *)0xE0001004;
		cyc[3]=cyc[3]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 4	Conv dw/ s2
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV4_IM_DIM, CONV4_IM_CH, conv4_wt, CONV4_OUT_CH, CONV4_KER_DIM,
												CONV4_PADDING, CONV4_STRIDE, conv4_bias, CONV4_BIAS_LSHIFT, CONV4_OUT_RSHIFT,
												img_buffer2, CONV4_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_4
		cyc[4]=*(volatile unsigned int *)0xE0001004;
		cyc[4]=cyc[4]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 5	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV5_IM_DIM, CONV5_IM_CH, conv5_wt, CONV5_OUT_CH, CONV5_KER_DIM,
								CONV5_PADDING, CONV5_STRIDE, conv5_bias, CONV5_BIAS_LSHIFT, CONV5_OUT_RSHIFT,
								img_buffer1, CONV5_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_5
		cyc[5]=*(volatile unsigned int *)0xE0001004;
		cyc[5]=cyc[5]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 6	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV6_IM_DIM, CONV6_IM_CH, conv6_wt, CONV6_OUT_CH, CONV6_KER_DIM,
												CONV6_PADDING, CONV6_STRIDE, conv6_bias, CONV6_BIAS_LSHIFT, CONV6_OUT_RSHIFT,
												img_buffer2, CONV6_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_6
		cyc[6]=*(volatile unsigned int *)0xE0001004;
		cyc[6]=cyc[6]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 7	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV7_IM_DIM, CONV7_IM_CH, conv7_wt, CONV7_OUT_CH, CONV7_KER_DIM,
								CONV7_PADDING, CONV7_STRIDE, conv7_bias, CONV7_BIAS_LSHIFT, CONV7_OUT_RSHIFT,
								img_buffer1, CONV7_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_7
		cyc[7]=*(volatile unsigned int *)0xE0001004;
		cyc[7]=cyc[7]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 8	Conv dw/ s2
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV8_IM_DIM, CONV8_IM_CH, conv8_wt, CONV8_OUT_CH, CONV8_KER_DIM,
												CONV8_PADDING, CONV8_STRIDE, conv8_bias, CONV8_BIAS_LSHIFT, CONV8_OUT_RSHIFT,
												img_buffer2, CONV8_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_8
		cyc[8]=*(volatile unsigned int *)0xE0001004;
		cyc[8]=cyc[8]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 9	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV9_IM_DIM, CONV9_IM_CH, conv9_wt, CONV9_OUT_CH, CONV9_KER_DIM,
								CONV9_PADDING, CONV9_STRIDE, conv9_bias, CONV9_BIAS_LSHIFT, CONV9_OUT_RSHIFT,
								img_buffer1, CONV9_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_9
		cyc[9]=*(volatile unsigned int *)0xE0001004;
		cyc[9]=cyc[9]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 10	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV10_IM_DIM, CONV10_IM_CH, conv10_wt, CONV10_OUT_CH, CONV10_KER_DIM,
												CONV10_PADDING, CONV10_STRIDE, conv10_bias, CONV10_BIAS_LSHIFT, CONV10_OUT_RSHIFT,
												img_buffer2, CONV10_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_10
		cyc[10]=*(volatile unsigned int *)0xE0001004;
		cyc[10]=cyc[10]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 11	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV11_IM_DIM, CONV11_IM_CH, conv11_wt, CONV11_OUT_CH, CONV11_KER_DIM,
								CONV11_PADDING, CONV11_STRIDE, conv11_bias, CONV11_BIAS_LSHIFT, CONV11_OUT_RSHIFT,
								img_buffer1, CONV11_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_11
		cyc[11]=*(volatile unsigned int *)0xE0001004;
		cyc[11]=cyc[11]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 12	Conv dw/ s2
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV12_IM_DIM, CONV12_IM_CH, conv12_wt, CONV12_OUT_CH, CONV12_KER_DIM,
												CONV12_PADDING, CONV12_STRIDE, conv12_bias, CONV12_BIAS_LSHIFT, CONV12_OUT_RSHIFT,
												img_buffer2, CONV12_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_12
		cyc[12]=*(volatile unsigned int *)0xE0001004;
		cyc[12]=cyc[12]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 13	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV13_IM_DIM, CONV13_IM_CH, conv13_wt, CONV13_OUT_CH, CONV13_KER_DIM,
								CONV13_PADDING, CONV13_STRIDE, conv13_bias, CONV13_BIAS_LSHIFT, CONV13_OUT_RSHIFT,
								img_buffer1, CONV13_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_13
		cyc[13]=*(volatile unsigned int *)0xE0001004;
		cyc[13]=cyc[13]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 14	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV14_IM_DIM, CONV14_IM_CH, conv14_wt, CONV14_OUT_CH, CONV14_KER_DIM,
												CONV14_PADDING, CONV14_STRIDE, conv14_bias, CONV14_BIAS_LSHIFT, CONV14_OUT_RSHIFT,
												img_buffer2, CONV14_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_14
		cyc[14]=*(volatile unsigned int *)0xE0001004;
		cyc[14]=cyc[14]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 15	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV15_IM_DIM, CONV15_IM_CH, conv15_wt, CONV15_OUT_CH, CONV15_KER_DIM,
								CONV15_PADDING, CONV15_STRIDE, conv15_bias, CONV15_BIAS_LSHIFT, CONV15_OUT_RSHIFT,
								img_buffer1, CONV15_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_15
		cyc[15]=*(volatile unsigned int *)0xE0001004;
		cyc[15]=cyc[15]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 16	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV16_IM_DIM, CONV16_IM_CH, conv16_wt, CONV16_OUT_CH, CONV16_KER_DIM,
												CONV16_PADDING, CONV16_STRIDE, conv16_bias, CONV16_BIAS_LSHIFT, CONV16_OUT_RSHIFT,
												img_buffer2, CONV16_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_16
		cyc[16]=*(volatile unsigned int *)0xE0001004;
		cyc[16]=cyc[16]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 17	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV17_IM_DIM, CONV17_IM_CH, conv17_wt, CONV17_OUT_CH, CONV17_KER_DIM,
								CONV17_PADDING, CONV17_STRIDE, conv17_bias, CONV17_BIAS_LSHIFT, CONV17_OUT_RSHIFT,
								img_buffer1, CONV17_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_17
		cyc[17]=*(volatile unsigned int *)0xE0001004;
		cyc[17]=cyc[17]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 18	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV18_IM_DIM, CONV18_IM_CH, conv18_wt, CONV18_OUT_CH, CONV18_KER_DIM,
												CONV18_PADDING, CONV18_STRIDE, conv18_bias, CONV18_BIAS_LSHIFT, CONV18_OUT_RSHIFT,
												img_buffer2, CONV18_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_18
		cyc[18]=*(volatile unsigned int *)0xE0001004;
		cyc[18]=cyc[18]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 19	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV19_IM_DIM, CONV19_IM_CH, conv19_wt, CONV19_OUT_CH, CONV19_KER_DIM,
								CONV19_PADDING, CONV19_STRIDE, conv19_bias, CONV19_BIAS_LSHIFT, CONV19_OUT_RSHIFT,
								img_buffer1, CONV19_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_19
		cyc[19]=*(volatile unsigned int *)0xE0001004;
		cyc[19]=cyc[19]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 20	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV20_IM_DIM, CONV20_IM_CH, conv20_wt, CONV20_OUT_CH, CONV20_KER_DIM,
												CONV20_PADDING, CONV20_STRIDE, conv20_bias, CONV20_BIAS_LSHIFT, CONV20_OUT_RSHIFT,
												img_buffer2, CONV20_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_20
		cyc[20]=*(volatile unsigned int *)0xE0001004;
		cyc[20]=cyc[20]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 21	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV21_IM_DIM, CONV21_IM_CH, conv21_wt, CONV21_OUT_CH, CONV21_KER_DIM,
								CONV21_PADDING, CONV21_STRIDE, conv21_bias, CONV21_BIAS_LSHIFT, CONV21_OUT_RSHIFT,
								img_buffer1, CONV21_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_21
		cyc[21]=*(volatile unsigned int *)0xE0001004;
		cyc[21]=cyc[21]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 22	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV22_IM_DIM, CONV22_IM_CH, conv22_wt, CONV22_OUT_CH, CONV22_KER_DIM,
												CONV22_PADDING, CONV22_STRIDE, conv22_bias, CONV22_BIAS_LSHIFT, CONV22_OUT_RSHIFT,
												img_buffer2, CONV22_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_22
		cyc[22]=*(volatile unsigned int *)0xE0001004;
		cyc[22]=cyc[22]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 23	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV23_IM_DIM, CONV23_IM_CH, conv23_wt, CONV23_OUT_CH, CONV23_KER_DIM,
								CONV23_PADDING, CONV23_STRIDE, conv23_bias, CONV23_BIAS_LSHIFT, CONV23_OUT_RSHIFT,
								img_buffer1, CONV23_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_23
		cyc[23]=*(volatile unsigned int *)0xE0001004;
		cyc[23]=cyc[23]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 24	Conv dw/ s2
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV24_IM_DIM, CONV24_IM_CH, conv24_wt, CONV24_OUT_CH, CONV24_KER_DIM,
												CONV24_PADDING, CONV24_STRIDE, conv24_bias, CONV24_BIAS_LSHIFT, CONV24_OUT_RSHIFT,
												img_buffer2, CONV24_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_24
		cyc[24]=*(volatile unsigned int *)0xE0001004;
		cyc[24]=cyc[24]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 25	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV25_IM_DIM, CONV25_IM_CH, conv25_wt, CONV25_OUT_CH, CONV25_KER_DIM,
								CONV25_PADDING, CONV25_STRIDE, conv25_bias, CONV25_BIAS_LSHIFT, CONV25_OUT_RSHIFT,
								img_buffer1, CONV25_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_25
		cyc[25]=*(volatile unsigned int *)0xE0001004;
		cyc[25]=cyc[25]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 26	Conv dw/ s1
	 *dephtwise
	 *Cycle =
	 */
	arm_depthwise_separable_conv_HWC_q7(img_buffer1, CONV26_IM_DIM, CONV26_IM_CH, conv26_wt, CONV26_OUT_CH, CONV26_KER_DIM,
												CONV26_PADDING, CONV26_STRIDE, conv26_bias, CONV26_BIAS_LSHIFT, CONV26_OUT_RSHIFT,
												img_buffer2, CONV26_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_26
		cyc[26]=*(volatile unsigned int *)0xE0001004;
		cyc[26]=cyc[26]-cyc[0];
		*DWT_CYCCNT = 0;
		//STOPWATCH_START
		cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 27	Conv Point/ s1
	 *pointwise
	 *Cycle =
	 */
	arm_convolve_HWC_q7_fast(img_buffer2, CONV27_IM_DIM, CONV27_IM_CH, conv27_wt, CONV27_OUT_CH, CONV27_KER_DIM,
								CONV27_PADDING, CONV27_STRIDE, conv27_bias, CONV27_BIAS_LSHIFT, CONV27_OUT_RSHIFT,
								img_buffer1, CONV27_OUT_DIM, (q15_t *) col_buffer, NULL);
		//STOPWATCH_STOP_27
		cyc[27]=*(volatile unsigned int *)0xE0001004;
		cyc[27]=cyc[27]-cyc[0];
	cyc[0]=*(volatile unsigned int *)0xE0001004;

	/*Layer 28	Avg Pool /s1
	 *Layer 29	FC /s1
	 *Layer 30	Softmax / s1
	 */
  return 0;
}
