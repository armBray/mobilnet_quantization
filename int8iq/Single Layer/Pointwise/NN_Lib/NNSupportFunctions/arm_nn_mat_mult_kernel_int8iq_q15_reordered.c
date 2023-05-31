/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_mat_mult_kernel_q7_q15_reordered.c
 * Description:  Matrix-multiplication function for convolution with reordered columns
 *
 * $Date:        17. January 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_math.h"

  /**
   * @brief Matrix-multiplication function for convolution with reordered columns
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always conssists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   *
   * @details
   *
   * This function assumes that data in pInBuffer are reordered
   */

uint8_t     *arm_nn_mat_mult_kernel_int8iq_q15_reordered(const uint8_t * pA, // MODIFICA q7_t --> uint8_t
                                                  const q15_t * pInBuffer,
												  const uint8_t zA, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
												  const uint8_t zB, // (parametro) MODIFICA
												  const uint8_t z, // (parametro) MODIFICA, MODIFICA q7_t --> uint8_t
												  const q15_t M_ZERO, // (parametro) MODIFICA
												  const q15_t n, // MODIFICA
                                                  const uint16_t ch_im_out,
                                                  const uint16_t numCol_A,
                                                  const int32_t * bias, // MODIFICA q7_t --> int32_t
												  uint8_t * pOut) // MODIFICA q7_t --> uint8_t
{

#if defined (ARM_MATH_DSP)
    /* set up the second output pointers */
	uint8_t     *pOut2 = pOut + ch_im_out; // +ch_im_out perché punta al pixel adiacente al primo
    int       i;

    /*
     * creo un vettore in cui si andranno a posizionare 2 valori:
	 * rispettivamente il valore di zA ripetuto 2 volte in modo tale
	 * poi da effettuare un casting a *__SIMD32() per renderlo compatibile
	 * con le operazioni successive
	 */


	 q15_t VzA[2] = {zA,zA};
	 const q15_t *pzA = VzA;
	 q31_t 		inzA = *__SIMD32(pzA);

	 q15_t VzB[2] = {zB,zB};
	 const q15_t *pzB = VzB;
	 q31_t 		inzB = *__SIMD32(pzB);

    /* this loop over rows in A */
    for (i = 0; i < ch_im_out; i += 2)
    {
        /* setup pointers for B */
        const q15_t *pB = pInBuffer; // puntatore alla prima cella di input
        const q15_t *pB2 = pB + numCol_A; // puntatore alla cella adiacente alla prima (input) perché viene aggiunto numColA ovvero il numero di canali di A

        /* align the second pointer for A */
        const uint8_t *pA2 = pA + numCol_A; //perché usa numCol_A?

        /* init the sum with bias */
        /*** MODIFICA ***/
        /*
         * inizialmente il calcolo della somma avveniva in questo modo:
         * q31_t     sum =  ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
         */
        q31_t     sum =  (q31_t)(bias[i]);
        q31_t     sum2 = (q31_t)(bias[i]);
        q31_t     sum3 = (q31_t)(bias[i + 1]);
        q31_t     sum4 = (q31_t)(bias[i + 1]);
        /*** MODIFICA ***/

        uint16_t  colCnt = numCol_A >> 2; // shift di 2 perché opera con 8 pesi per ciclo (ricorda che in inA11 sono presenti 2 pesi) e con 2 pixel adiacenti per ciclo (inB1, inB2)
        /* accumulate over the vector */
        while (colCnt)
        {
            q31_t     inA11, inA12, inA21, inA22;
            /*
             * N.B.: in ogni variabile sono memorizzati 2 pesi attraverso la funzione read_and_pad_reordered()
             * Esempio: inA1 : | peso1 | peso 2 |
             * ---> vedi pag. 3 PAPER_1[1]
             * ---> vedi ALLEGATO PAPER
             */
            q31_t     inB1 = *__SIMD32(pB)++; //per capire cosa è stato attribuito basta guardare la più/meno cifra significativa in HEX
            q31_t     inB2 = *__SIMD32(pB2)++;

            /*
             * Con la seguente funzione il vettore pA (q7_t) viene selezionato;
             * vengono dunque considerati 4 valori a partire dal suo puntatore e viene effettuata
             * una estensione a 16 bit di cianscun valore.
             * si ottengono dunque 2 vettori da q31_t (inA11, inA12 or inA21, inA22),
             * ciascuno dei quali contiene 2 valori da q15_t, ovvero 2 pesi  da q15_t
             * ---> vedi pag. 3 PAPER_1[1]
             * ---> vedi ALLEGATO PAPER
             *
             * inoltre come viene spiegato nel penultimo paragrafo in pag. 3 PAPER_1[1],
             * l'operazione di riordino non è necessaria perché pB (ovvero l'indirizzo del vettore dei pixel)
             * ha subito la stessa operazione di estensione!
             */
            pA = (uint8_t *) read_and_pad_int8iq_reordered((void *)pA, &inA11, &inA12);
            pA2 = (uint8_t *) read_and_pad_int8iq_reordered((void *)pA2, &inA21, &inA22);

            /*** MODIFICA ***/
            /* Modifica pA*/

             inA11 = __SSUB16(inA11, inzA); //NB il tipo di variabile di inzA è uint32_t perché la SSUB richiede quello!
             inA12 = __SSUB16(inA12, inzA);
             inA21 = __SSUB16(inA21, inzA);
             inA22 = __SSUB16(inA22, inzA);

            /* Modifica pB*/
             inB1 = __SSUB16(inB1, inzB);
             inB2 = __SSUB16(inB2, inzB);

            /*** MODIFICA **_OK_*/

            sum = __SMLAD(inA11, inB1, sum);
            sum2 = __SMLAD(inA11, inB2, sum2);
            sum3 = __SMLAD(inA21, inB1, sum3);
            sum4 = __SMLAD(inA21, inB2, sum4);

            inB1 = *__SIMD32(pB)++;
            inB2 = *__SIMD32(pB2)++;

            /*** MODIFICA ***/
            /* Modifica pB*/
            inB1 = __SSUB16(inB1, inzB);
            inB2 = __SSUB16(inB2, inzB);

			/*** MODIFICA **_OK_*/

            sum = __SMLAD(inA12, inB1, sum);
            sum2 = __SMLAD(inA12, inB2, sum2);
            sum3 = __SMLAD(inA22, inB1, sum3);
            sum4 = __SMLAD(inA22, inB2, sum4);

            colCnt--;
        }                       /* while over colCnt */


        //*** TO BE TESTED ***
        colCnt = numCol_A & 0x3; // bitwise operator: controlla se numCol_A è multiplo di 4 o meno
        while (colCnt)
        {
            int16_t      inA1 = (int16_t)*pA++; // MODIFICA inizialmente era q7_t  inA1 = *pA++;
            q15_t     inB1 = *pB++;
            int16_t      inA2 = (int16_t)*pA2++;
            q15_t     inB2 = *pB2++;

            /*** MODIFICA ***/
            /*
             * qui si mette VzA perché NON viene utilizzata la funzione di estensione
             */
			 inA1 = inA1 - VzA[0];
			 inB1 = inB1 - VzB[0];
			 inA2 = inA2 - VzA[0];
             inB2 = inB2 - VzB[0];

			/*** MODIFICA **_OK_*/

            sum += inA1 * inB1;
            sum2 += inA1 * inB2;
            sum3 += inA2 * inB1;
            sum4 += inA2 * inB2;
            colCnt--;
        }                       /* while over colCnt */

        /*** MODIFICA ***/
		/*
		 * la somma finale viene moltiplicata per M=M_0*2^(-n)
		 */
		 sum = ((sum*M_ZERO) >> n) + z;
		 sum2 = ((sum2*M_ZERO) >> n) + z;
		 sum3 = ((sum3*M_ZERO) >> n) + z;
		 sum4 = ((sum4*M_ZERO) >> n) + z;

		/*** MODIFICA **_OK_*/
        /*
         * inizialmente era così: *pOut++ = (q7_t) __SSAT((sum >> out_shift), 8);
         * è stata fatta la modifica visto lo shift effettuato precedentemente con >> n
		 * N.B.: le assegnazioni di sum in pout sono invertite
		 * perché in sum & sum3 sono presenti i valori relativi a pA
		 * mentre in sum2 & sum4 sono presenti i valori relativi a pA2
         */
        *pOut++ = (uint8_t) __USAT(sum, 8);
        *pOut++ = (uint8_t) __USAT(sum3, 8);
        *pOut2++ = (uint8_t) __USAT(sum2, 8);
        *pOut2++ = (uint8_t) __USAT(sum4, 8);

        /* skip the row computed with A2 */
        pA += numCol_A;
    }                           /* for over ch_im_out */

    pOut += ch_im_out;

    /* return the new output pointer with offset */
    return pOut;
#else
    /* To be completed */
    return NULL;
#endif                          /* ARM_MATH_DSP */
}
