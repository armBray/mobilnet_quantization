/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_1x1_HWC_in8iq_fast_nonsquare.c
 * Description:  Fast int8iq version of 1x1 convolution (non-square shape)
 *
 * $Date:        December 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Fast Q7 version of 1x1 convolution (non-sqaure shape)
 * @param[in]       Im_in        pointer to input tensor
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       dim_im_in_y  input tensor dimention y
 * @param[in]       ch_im_in     number of input tensor channels
 * @param[in]       wt           pointer to kernel weights
 * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x filter kernel size x
 * @param[in]       dim_kernel_y filter kernel size y
 * @param[in]       padding_x    padding size x
 * @param[in]       padding_y    padding size y
 * @param[in]       stride_x     convolution stride x
 * @param[in]       stride_y     convolution stride y
 * @param[in]       bias         pointer to bias
 * @param[in]       bias_shift   amount of left-shift for bias
 * @param[in]       out_shift    amount of right-shift for output
 * @param[in,out]   Im_out       pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in]       dim_im_out_y output tensor dimension y
 * @param[in,out]   bufferA      pointer to buffer space for input 
 * @param[in,out]   bufferB      pointer to buffer space for output
 * @return     The function returns either
 * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
 *
 * This function is optimized for convolution with 1x1 kernel size (i.e., dim_kernel_x=1
 * and dim_kernel_y=1). It can be used for the second half of MobileNets [1] after depthwise 
 * separable convolution.
 *
 * This function is the version with full list of optimization tricks, but with
 * some contraints:
 *   ch_im_in is multiple of 4
 *   ch_im_out is multiple of 2
 *
 * [1] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
 * https://arxiv.org/abs/1704.04861
 */

arm_status arm_convolve_1x1_HWC_int8iq_fast_nonsquare(const uint8_t * Im_in,
                                                  const uint16_t dim_im_in_x,
                                                  const uint16_t dim_im_in_y,
                                                  const uint16_t ch_im_in,
                                                  const uint8_t * wt,
                                                  const uint8_t zA, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
												  const uint8_t zB, // (parametro) MODIFICA
												  const uint8_t z, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
												  const q15_t M_ZERO, // (parametro) MODIFICA
												  const q15_t n, // (parametro)MODIFICA
                                                  const uint16_t ch_im_out,
                                                  const uint16_t dim_kernel_x,
                                                  const uint16_t dim_kernel_y,
                                                  const uint16_t padding_x,
                                                  const uint16_t padding_y,
                                                  const uint16_t stride_x,
                                                  const uint16_t stride_y,
                                                  const int32_t * bias,
                                                  uint8_t * Im_out,
                                                  const uint16_t dim_im_out_x,
                                                  const uint16_t dim_im_out_y, 
                                                  q15_t * bufferA, 
                                                  q7_t * bufferB)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_out_y, i_out_x;
    int16_t   i_ch_out;

    /* -----------------------
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q15_t    *pBuffer = bufferA;
    q7_t     *pOut = Im_out;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0 || dim_kernel_x != 1 || dim_kernel_y != 1
        || padding_x != 0 || padding_y != 0 || stride_x != 1 || stride_y != 1)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
        {
            /* This part implements the im2col function */
            arm_int8iq_to_q15_reordered_no_shift((q7_t *) Im_in + (i_out_y * dim_im_in_x + i_out_x) * ch_im_in, pBuffer,
                                             ch_im_in);
            pBuffer += ch_im_in;

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int8iq_q15_reordered(wt, bufferA,zA,
															zB,
															z,
															M_ZERO,
															n, ch_im_out, ch_im_in, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        const q7_t *pA = wt;

        // creo un vettore in cui si andranno a posizionare 2 valori:
			//rispettivamente il valore di zA ripetuto 2 volte

		 q15_t VzA[2] = {zA,zA};
		 const q15_t *pzA = VzA;
		 q31_t 		inzA = *__SIMD32(pzA);

		 q15_t VzB[2] = {zB,zB};
		 const q15_t *pzB = VzB;
		 q31_t 		inzB = *__SIMD32(pzB);

        for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
        {
            q31_t     sum = (q31_t)(bias[i_ch_out]);
            q15_t    *pB = bufferA;
            /* basically each time it process 4 entries */
            uint16_t  colCnt = ch_im_in * dim_kernel_x * dim_kernel_y >> 2;

            while (colCnt)
            {

                q31_t     inA1, inA2;
                q31_t     inB1, inB2;

                pA = (const q7_t *)read_and_pad_int8iq_reordered((void *)pA, &inA1, &inA2);

                inB1 = *__SIMD32(pB)++;

                /*** MODIFICA ***/
			   /* Modifica pA*/

				inA1 = __SSUB16(inA1, inzA); //NB il tipo di variabile di inzA è uint32_t perché la SSUB richiede quello!
				inA2 = __SSUB16(inA2, inzA);

			   /* Modifica pB*/
				inB1 = __SSUB16(inB1, inzB);
                sum = __SMLAD(inA1, inB1, sum);

                inB2 = *__SIMD32(pB)++;
                /*** MODIFICA ***/
				/* Modifica pB*/
			   inB2 = __SSUB16(inB2, inzB);
                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = ch_im_in * dim_kernel_y * dim_kernel_x & 0x3;
            while (colCnt)
            {
                q7_t      inA1 = *pA++;
                q15_t     inB1 = *pB++;
                sum += inA1 * inB1;
                colCnt--;
            }
            /*
			 * la somma finale viene moltiplicata per M=M_0*2^(-n)
			 */
			 sum = ((sum*M_ZERO) >> n) + z;

			*pOut = (uint8_t) __USAT(sum , 8);
            pOut++;

        }

    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
		
    int       i, j, k, l, m, n;
    int       conv_out;
    int       in_row, in_col;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0 || dim_kernel_x != 1 || dim_kernel_y != 1
        || padding_x != 0 || padding_y != 0 || stride_x != 1 || stride_y != 1)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
                conv_out = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_x; n++)
                    {
                        // if-for implementation
                        in_row = stride_y * j + m - padding_y;
                        in_col = stride_x * k + n - padding_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + l] *
                                    wt[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_y + n) * ch_im_in + l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t) __SSAT((conv_out >> out_shift), 8);
            }
        }
    }

#endif                          /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
