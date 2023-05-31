/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_HWC_int8iq_fast.c
 * Description:  Fast int8iq version of convolution
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
   * @brief Fast Q7 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimention
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input 
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
   *
   * bufferB size: 0
   *
   * <b>Input dimension constraints:</b>
   *
   * ch_im_in is multiple of 4    ( because of the SIMD32 read and swap )
   *
   * ch_im_out is multipe of 2    ( bacause 2x2 mat_mult kernel )
   *
   * The im2col converts the Q7 tensor input into Q15 column, which is stored in
   * bufferA. There is reordering happenning during this im2col process with
   * arm_q7_to_q15_reordered_no_shift. For every four elements, the second and
   * third elements are swapped. 
   *
   * The computation kernel arm_nn_mat_mult_kernel_q7_q15_reordered does the
   * GEMM computation with the reordered columns.
   *
   * To speed-up the determination of the padding condition, we split the
   * computation into 3x3 parts, i.e., {top, mid, bottom} X {left, mid, right}.
   * This reduces the total number of boundary condition checks and improves
   * the data copying performance.
   */

arm_status
arm_convolve_HWC_int8iq_fast(const uint8_t * Im_in, // MODIFICA q7_t --> uint8_t
                         const uint16_t dim_im_in,
                         const uint16_t ch_im_in,
                         const uint8_t * wt, // (buffer) MODIFICA q7_t --> uint8_t
						 const uint8_t zA, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
						 const uint8_t zB, // (parametro) MODIFICA
						 const uint8_t z, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
						 const q15_t M_ZERO, // (parametro) MODIFICA
						 const q15_t n, // (parametro)MODIFICA
                         const uint16_t ch_im_out,
                         const uint16_t dim_kernel,
                         const uint16_t padding,
                         const uint16_t stride,
                         const int32_t * bias, // MODIFICA q7_t --> int32_t
                         uint8_t * Im_out, // MODIFICA q7_t --> uint8_t
                         const uint16_t dim_im_out, 
                         q15_t * bufferA, 
                         q7_t * bufferB)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_out_y, i_out_x, i_ker_y, i_ker_x;

    /*
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q15_t    *pBuffer = bufferA;
    uint8_t     *pOut = Im_out;


    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    /*
     *  Here we split the entire matrix into three regions depending on the padding situation
     *    Top: i_out_y from 0 to padding - 1
     * Middle: i_out_y from padding to dim_im_out-padding-1
     * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
     */

    /* top part */
    for (i_out_y = 0; i_out_y < padding; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                    	/*
                    	 * riempie di 0 se previsto padding
                    	 */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                    	/*
                    	 * Questa funzione opera sul buffer di Input:
                    	 *  1) preleva i valori di Im_in in formato q7_t;
                    	 *  2) li espande da q7_7 a q15_t;
                    	 *  3) li memorizza in pBuffer.
                    	 *
                    	 ***** N.B.: l'operazione di espansione viene effettuata solamente per gli input!
                    	 ***** per quanto riguarda i pesi questi verranno espansi all'interno della funzione
                    	 ***** arm_nn_mat_mult_kernel_int8iq_q15_reordered(), attraverso la funzione read_and_pad_reordered().
                    	 */
                    	arm_int8iq_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int8iq_q15_reordered(wt,
                                                            bufferA,
															zA,
															zB,
															z,
															M_ZERO,
															n,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* middle part, here we also divide the x into left, mid and right */
    for (; i_out_y < dim_im_out - padding; i_out_y++)
    {

        /* left part */
        for (i_out_x = 0; i_out_x < padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                    	arm_int8iq_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int8iq_q15_reordered(wt,
                                                            bufferA,
															zA,
															zB,
															z,
															M_ZERO,
															n,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* mid part */
        for (; i_out_x < dim_im_out - padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
            	arm_int8iq_to_q15_reordered_no_shift((q7_t *) Im_in
                                                 +
                                                 (i_ker_y *
                                                  dim_im_in +
                                                  i_out_x *
                                                  stride - padding) * ch_im_in, pBuffer, ch_im_in * dim_kernel);
                pBuffer += ch_im_in * dim_kernel;
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int8iq_q15_reordered(wt,
                                                            bufferA,
															zA,
															zB,
															z,
															M_ZERO,
															n,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* right part */
        for (; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                    	arm_int8iq_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut =
                    arm_nn_mat_mult_kernel_int8iq_q15_reordered(wt,
                                                            bufferA,
															zA,
															zB,
															z,
															M_ZERO,
															n,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    for (; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t)*ch_im_in);
                    } else
                    {
                    	arm_int8iq_to_q15_reordered_no_shift
                            ((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
            	/*** MODIFICA ***/
                pOut =
                    arm_nn_mat_mult_kernel_int8iq_q15_reordered(wt,
                                                            bufferA,
															zA,
															zB,
															z,
															M_ZERO,
															n,
                                                            ch_im_out,
                                                            ch_im_in
                                                            *
                                                            dim_kernel * dim_kernel, bias, pOut);
                /*** MODIFICA ***/

                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /***********
     *********** N.B.: questo ultimo loop non è stato modificato
     *********** quindi sarà del tutto normale avere l'ultimo pixel non corrispondente!!!
     */
    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        const uint8_t *pA = wt;
        int       i;

        // creo un vettore in cui si andranno a posizionare 2 valori:
        	//rispettivamente il valore di zA ripetuto 2 volte

		 q15_t VzA[2] = {zA,zA};
		 const q15_t *pzA = VzA;
		 q31_t 		inzA = *__SIMD32(pzA);

		 q15_t VzB[2] = {zB,zB};
		 const q15_t *pzB = VzB;
		 q31_t 		inzB = *__SIMD32(pzB);

        for (i = 0; i < ch_im_out; i++)
        {
            q31_t     sum = (q31_t)bias[i];
            q15_t    *pB = bufferA;
            /* each time it process 4 entries */
            uint16_t  colCnt = ch_im_in * dim_kernel * dim_kernel >> 2;

            while (colCnt)
            {

                q31_t     inA1, inA2;
                q31_t     inB1, inB2;

                pA = (uint8_t *) read_and_pad_int8iq_reordered((void *)pA, &inA1, &inA2);


                inB1 = *__SIMD32(pB)++;

                /*** MODIFICA ***/
			   /* Modifica pA*/

				inA1 = __SSUB16(inA1, inzA); //NB il tipo di variabile di inzA è uint32_t perché la SSUB richiede quello!
				inA2 = __SSUB16(inA2, inzA);

			   /* Modifica pB*/
				inB1 = __SSUB16(inB1, inzB);

			   /*** MODIFICA **_OK_*/

                sum = __SMLAD(inA1, inB1, sum);

                inB2 = *__SIMD32(pB)++;
                /*** MODIFICA ***/
				/* Modifica pB*/
                inB2 = __SSUB16(inB2, inzB);

				/*** MODIFICA **_OK_*/
                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = ch_im_in * dim_kernel * dim_kernel & 0x3;
            while (colCnt)
            {
            	int16_t      inA1 = (int16_t)*pA++;
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

    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    signed char in_row, in_col;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out; j++)
        {
            for (k = 0; k < dim_im_out; k++)
            {
                conv_out = (bias[i] << bias_shift) + NN_ROUND(out_shift);
                for (m = 0; m < dim_kernel; m++)
                {
                    for (n = 0; n < dim_kernel; n++)
                    {
                        // if-for implementation
                        in_row = stride * j + m - padding;
                        in_col = stride * k + n - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out +=
                                    Im_in[(in_row * dim_im_in + in_col) * ch_im_in +
                                          l] * wt[i * ch_im_in * dim_kernel * dim_kernel + (m * dim_kernel +
                                                                                            n) * ch_im_in + l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out + k) * ch_im_out] = (q7_t) __SSAT((conv_out >> out_shift), 8);
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
