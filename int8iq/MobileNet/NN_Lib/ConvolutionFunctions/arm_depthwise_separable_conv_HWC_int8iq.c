/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_depthwise_separable_conv_HWC_q7.c
 * Description:  Q7 depthwise separable convolution function
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
 * @brief Q7 depthwise separable convolution function
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
 * ch_im_in equals ch_im_out
 *
 * Implementation:
 * There are 3 nested loop here:
 * Inner loop: calculate each output value with MAC instruction over an accumulator
 * Mid   loop: loop over different output channel
 * Outer loop: loop over different output (x, y)
 */

arm_status arm_depthwise_separable_conv_HWC_int8iq(const uint8_t * Im_in,
											   const uint16_t dim_im_in,
											   const uint16_t ch_im_in,
											   const uint8_t * wt,
											   const uint8_t zA, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
											   const uint8_t zB, // (parametro) MODIFICA
											   const uint8_t z, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
											   const q15_t M_ZERO, // (parametro) MODIFICA
											   const q15_t n, // (parametro) MODIFICA
											   const uint16_t ch_im_out,
											   const uint16_t dim_kernel,
											   const uint16_t padding,
											   const uint16_t stride,
											   const int32_t * bias,
											   //const uint16_t bias_shift,
											   //const uint16_t out_shift,
											   uint8_t * Im_out,
											   const uint16_t dim_im_out,
											   q15_t * bufferA,
											   q7_t * bufferB)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t   i_out_y, i_out_x;
    int16_t   i_ker_y, i_ker_x;
    q7_t     *colBuffer = (q7_t *) bufferA;
    q7_t     *pBuffer = colBuffer;
    const int32_t *pBias = bias;
    uint8_t     *pOut = Im_out;
    uint16_t  rowCnt;
    uint16_t  row_shift;

    q15_t VzA[2] = {zA,zA};
	const q15_t *pzA = VzA;
	q31_t 		inzA = *__SIMD32(pzA);

	q15_t VzB[2] = {zB,zB};
	const q15_t *pzB = VzB;
	q31_t 		inzB = *__SIMD32(pzB);

    /* do some checking here, basically ch_im_in == ch_im_out */
    if (ch_im_in != ch_im_out)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* we first do im2col here */
            for (i_ker_y = i_out_y * stride - padding; i_ker_y < i_out_y * stride - padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - padding; i_ker_x < i_out_x * stride - padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        /* arm_fill_q7(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, ch_im_in);
                    } else
                    {
                        /* arm_copy_q7((q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer, ch_im_in); */
                        memcpy(pBuffer, (q7_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            /* we will do the computation here for each channel */
            rowCnt = ch_im_out >> 2;
            row_shift = 0;
            pBias = bias;

            while (rowCnt)
            {
                q31_t     sum =  (q31_t)(*pBias++); //q31_t     sum =  ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
                q31_t     sum2 = (q31_t)(*pBias++);
                q31_t     sum3 = (q31_t)(*pBias++);
                q31_t     sum4 = (q31_t)(*pBias++);

                uint16_t  colCnt = (dim_kernel * dim_kernel) >> 1;
                q7_t     *pB = colBuffer + row_shift;
                const uint8_t *pA = wt + row_shift;
                row_shift += 4;

#ifdef USE_INTRINSIC

#ifndef ARM_MATH_BIG_ENDIAN

                while (colCnt)
                {
                    q31_t     inA1, inA2, inB1, inB2, opA, opB;

                    inB1 = *__SIMD32(pB);
                    pB += ch_im_in;
                    opB = *__SIMD32(pB);
                    pB += ch_im_in;
                    inB2 = __PKHTB(opB, inB1, 16); //uint32_t __PKHTB	(uint32_t val1, uint32_t val2, uint32_t val3)
                    inB1 = __PKHBT(inB1, opB, 16); //uint32_t __PKHBT	(uint32_t val1, uint32_t val2, uint32_t val3)
                    inA1 = *__SIMD32(pA);
                    pA += ch_im_in;
                    opB = *__SIMD32(pA);
                    pA += ch_im_in;
                    inA2 = __PKHTB(opB, inA1, 16);
                    inA1 = __PKHBT(inA1, opB, 16);

                    //sum
                    opA = __UXTB16(inA1); //opA = __SXTB16(inA1);
                    opB = __UXTB16(inB1);

                    opA = __SSUB16(opA, inzA);
                    opB = __SSUB16(opB, inzB);
                    sum = __SMLAD(opA, opB, sum);

                    //sum2
                    opA = __UXTB16(__ROR(inA1, 8));
                    opB = __UXTB16(__ROR(inB1, 8));

                    opA = __SSUB16(opA, inzA);
					opB = __SSUB16(opB, inzB);
                    sum2 = __SMLAD(opA, opB, sum2);

                    //sum3
                    opA = __UXTB16(inA2);
                    opB = __UXTB16(inB2);

                    opA = __SSUB16(opA, inzA);
					opB = __SSUB16(opB, inzB);
                    sum3 = __SMLAD(opA, opB, sum3);

                    //sum4
                    opA = __UXTB16(__ROR(inA2, 8));
                    opB = __UXTB16(__ROR(inB2, 8));

                    opA = __SSUB16(opA, inzA);
					opB = __SSUB16(opB, inzB);
                    sum4 = __SMLAD(opA, opB, sum4);
                    colCnt--;
                }
#else

                while (colCnt)
                {
                    q31_t     inA1, inA2, inB1, inB2, opA, opB;

                    inB1 = *__SIMD32(pB);
                    pB += ch_im_in;
                    opB = *__SIMD32(pB);
                    pB += ch_im_in;
                    inB2 = __PKHBT(opB, inB1, 16);
                    inB1 = __PKHTB(inB1, opB, 16);
                    inA1 = *__SIMD32(pA);
                    pA += ch_im_in;
                    opB = *__SIMD32(pA);
                    pA += ch_im_in;
                    inA2 = __PKHBT(opB, inA1, 16);
                    inA1 = __PKHTB(inA1, opB, 16);
                    opA = __SXTB16(inA1);
                    opB = __SXTB16(inB1);
                    sum2 = __SMLAD(opA, opB, sum2);
                    opA = __SXTB16(__ROR(inA1, 8));
                    opB = __SXTB16(__ROR(inB1, 8));
                    sum = __SMLAD(opA, opB, sum);
                    opA = __SXTB16(inA2);
                    opB = __SXTB16(inB2);
                    sum4 = __SMLAD(opA, opB, sum4);
                    opA = __SXTB16(__ROR(inA2, 8));
                    opB = __SXTB16(__ROR(inB2, 8));
                    sum3 = __SMLAD(opA, opB, sum3);
                    colCnt--;
                }

#endif                          /* ARM_MATH_BIG_ENDIAN */

#else

#ifndef ARM_MATH_BIG_ENDIAN
                /*
                 *   r0    r1    r2    r3    r4   r5
                 *  inA1, inA2, inB1, inB2, opA, opB
                 */

                asm volatile ("COL_LOOP_%=:\n"
                              "ldr.w r2, [%[pB], #0]\n"
                              "add.w %[pB], %[pB], %[ch_im_in]\n"
                              "ldr.w r5, [%[pB], #0]\n"
                              "add.w %[pB], %[pB], %[ch_im_in]\n"
                              "pkhtb r3, r5, r2, ASR #16\n"
                              "pkhbt r2, r2, r5, LSL #16\n"
                              "ldr.w r0, [%[pA], #0]\n"
                              "add.w %[pA], %[pA], %[ch_im_in]\n"
                              "ldr.w r5, [%[pA], #0]\n"
                              "add.w %[pA], %[pA], %[ch_im_in]\n"
                              "pkhtb r1, r5, r0, ASR #16\n"
                              "pkhbt r0, r0, r5, LSL #16\n"
                              "sxtb16 r4, r0\n"
                              "sxtb16 r5, r2\n"
                              "smlad %[sum], r4, r5, %[sum]\n"
                              "mov.w r4, r0, ror #8\n"
                              "mov.w r5, r2, ror #8\n"
                              "sxtb16 r4, r4\n"
                              "sxtb16 r5, r5\n"
                              "smlad %[sum2], r4, r5, %[sum2]\n"
                              "sxtb16 r4, r1\n"
                              "sxtb16 r5, r3\n"
                              "smlad %[sum3], r4, r5, %[sum3]\n"
                              "mov.w r4, r1, ror #8\n"
                              "mov.w r5, r3, ror #8\n"
                              "sxtb16 r4, r4\n"
                              "sxtb16 r5, r5\n"
                              "smlad %[sum4], r4, r5, %[sum4]\n"
                              "subs %[colCnt], #1\n"
                              "bne COL_LOOP_%=\n":[sum]
                              "+r"(sum),[sum2] "+r"(sum2),
                              [sum3] "+r"(sum3),
                              [sum4] "+r"(sum4),[pB] "+r"(pB),
                              [pA] "+r"(pA):[colCnt]
                              "r"(colCnt),[ch_im_in] "r"(ch_im_in):"r0", "r1", "r2", "r3", "r4", "r5");
#else
                /*
                 *  r0    r1    r2    r3    r4   r5
                 * inA1, inA2, inB1, inB2, opA, opB
                 */
                asm volatile ("COL_LOOP_%=:\n"
                              "ldr.w r2, [%[pB], #0]\n"
                              "add.w %[pB], %[pB], %[ch_im_in]\n"
                              "ldr.w r5, [%[pB], #0]\n"
                              "add.w %[pB], %[pB], %[ch_im_in]\n"
                              "pkhbt r3, r5, r2, LSL #16\n"
                              "pkhtb r2, r2, r5, ASR #16\n"
                              "ldr.w r0, [%[pA], #0]\n"
                              "add.w %[pA], %[pA], %[ch_im_in]\n"
                              "ldr.w r5, [%[pA], #0]\n"
                              "add.w %[pA], %[pA], %[ch_im_in]\n"
                              "pkhbt r1, r5, r0, LSL #16\n"
                              "pkhtb r0, r0, r5, ASR #16\n"
                              "sxtb16 r4, r0\n"
                              "sxtb16 r5, r2\n"
                              "smlad %[sum2], r4, r5, %[sum2]\n"
                              "mov.w r4, r0, ror #8\n"
                              "mov.w r5, r2, ror #8\n"
                              "sxtb16 r4, r4\n"
                              "sxtb16 r5, r5\n"
                              "smlad %[sum], r4, r5, %[sum]\n"
                              "sxtb16 r4, r1\n"
                              "sxtb16 r5, r3\n"
                              "smlad %[sum4], r4, r5, %[sum4]\n"
                              "mov.w r4, r1, ror #8\n"
                              "mov.w r5, r3, ror #8\n"
                              "sxtb16 r4, r4\n"
                              "sxtb16 r5, r5\n"
                              "smlad %[sum3], r4, r5, %[sum3]\n"
                              "subs %[colCnt], #1\n"
                              "bne COL_LOOP_%=\n":[sum]
                              "+r"(sum),[sum2] "+r"(sum2),
                              [sum3] "+r"(sum3),
                              [sum4] "+r"(sum4),[pB] "+r"(pB),
                              [pA] "+r"(pA):[colCnt]
                              "r"(colCnt),[ch_im_in] "r"(ch_im_in):"r0", "r1", "r2", "r3", "r4", "r5");

#endif                          /* ARM_MATH_BIG_ENDIAN */

#endif                          /* USE_INTRINSIC */


                /*
                 * TO BE MODIFIED!!!!!
                 */
                colCnt = (dim_kernel * dim_kernel) & 0x1;
                while (colCnt)
                {
                    union arm_nnword inA, inB;
                    inA.word = *__SIMD32(pA);
                    pA += ch_im_in;
                    inB.word = *__SIMD32(pB);
                    pB += ch_im_in;

                    /*
                     * aggiungi quì sottrazione
                     */

                    sum += inA.bytes[0] * inB.bytes[0];
                    sum2 += inA.bytes[1] * inB.bytes[1];
                    sum3 += inA.bytes[2] * inB.bytes[2];
                    sum4 += inA.bytes[3] * inB.bytes[3];
                    colCnt--;
                }

                /*** MODIFICA ***/
				/*
				 * la somma finale viene moltiplicata per M=M_0*2^(-n)
				 */
				 sum = ((sum*M_ZERO) >> n) + z;
				 sum2 = ((sum2*M_ZERO) >> n) + z;
				 sum3 = ((sum3*M_ZERO) >> n) + z;
				 sum4 = ((sum4*M_ZERO) >> n) + z;

                *pOut++ = (uint8_t) __USAT(sum, 8);
                *pOut++ = (uint8_t) __USAT(sum2, 8);
                *pOut++ = (uint8_t) __USAT(sum3, 8);
                *pOut++ = (uint8_t) __USAT(sum4, 8);

                rowCnt--;
            }

            rowCnt = ch_im_out & 0x3;
            while (rowCnt)
            {
                q7_t     *pB = colBuffer + row_shift;
                const uint8_t *pA = wt + row_shift;
                q31_t     sum = (q31_t)(*pBias++);
                uint16_t  colCnt = (dim_kernel * dim_kernel);

                row_shift += 1;

                while (colCnt)
                {
                    q7_t      A1 = *pA;
                    q7_t      B1 = *pB;
                    pA += ch_im_in;
                    pB += ch_im_in;

                    /*
					 * aggiungi quì sottrazione
					 */

                    sum += A1 * B1;

                    colCnt--;
                }
                *pOut++ = (uint8_t) __USAT(sum, 8);
                rowCnt--;
            }

            /* clear counter and pointers */
            pBuffer = colBuffer;
        }
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    int       i_out_y, i_out_x, i_ch_out, i_ker_x, i_ker_y;
    int       conv_out;

    /* do some checking here, basically ch_im_in == ch_im_out */
    if (ch_im_in != ch_im_out)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }

    for (i_out_y = 0; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
            {
                // for each output
                conv_out = ((q31_t)(bias[i_ch_out]) << bias_shift) + NN_ROUND(out_shift);
                for (i_ker_y = 0; i_ker_y < dim_kernel; i_ker_y++)
                {
                    for (i_ker_x = 0; i_ker_x < dim_kernel; i_ker_x++)
                    {
                        int       in_row = stride * i_out_y + i_ker_y - padding;
                        int       in_col = stride * i_out_x + i_ker_x - padding;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in && in_col < dim_im_in)
                        {
                            conv_out +=
                                Im_in[(in_row *
                                       dim_im_in +
                                       in_col) *
                                      ch_im_in +
                                      i_ch_out] * wt[(i_ker_y * dim_kernel + i_ker_x) * ch_im_out + i_ch_out];
                        }
                    }
                }
                Im_out[(i_out_y * dim_im_out +
                        i_out_x) * ch_im_out + i_ch_out] = (q7_t) __SSAT((conv_out >> out_shift), 8);
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
