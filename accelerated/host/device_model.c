/**
 * device_model.c - Bit-exact software model of the systolic array's block matmul
 */

#include "device_model.h"

#include <stddef.h>
#include <stdint.h>

#include "config.h"

// float16 field geometry the array is elaborated with, in mac.sv's terms
#define MANT_WIDTH (F16_FRAC_BITS + 1)   // 11, including the hidden bit
#define PROD_WIDTH (2 * MANT_WIDTH)      // 22, exact

// the mantissa product's LSB has a weight of
// 2**(exp_a + exp_b - SHIFT_BIAS - ACC_LSB)
#define SHIFT_BIAS (2 * F16_BIAS + 2 * F16_FRAC_BITS + ACC_LSB)

#define ACC_MASK (((uint64_t)1 << ACC_WIDTH) - 1)
#define ACC_SIGN ((uint64_t)1 << (ACC_WIDTH - 1))

// one bit wider than the accumulator so that negating the most negative value
// does not alias back onto itself (fix_to_float.sv's MAG_WIDTH)
#define MAG_WIDTH (ACC_WIDTH + 1)

// a magnitude whose MSB sits at bit index `i` has a value of 2**(i + ACC_LSB),
// so this is what turns that index into a biased exponent
#define EXP_OFFSET (ACC_LSB + F16_BIAS)

// width of the exponent field, and the all-ones value it reserves
#define F16_EXP_BITS      (16 - 1 - F16_FRAC_BITS)      // 5
#define F16_EXP_FIELD_MAX ((1 << F16_EXP_BITS) - 1)     // 31

/**
 * @brief Wraps a value into the accumulator's width, as the RTL's adder does
 *
 * @param[in] v Value to wrap
 * @return      `v` reduced mod 2**ACC_WIDTH, interpreted as signed
 */
static inline int64_t acc_trunc(int64_t v) {
    uint64_t u = (uint64_t)v & ACC_MASK;

    return (int64_t)((u ^ ACC_SIGN) - ACC_SIGN);
}

/**
 * @brief One multiply-accumulate, mirroring mac.sv
 *
 * The mantissa product is exact; only the alignment shift discards bits, and
 * only below 2**ACC_LSB. Subnormal operands flush to zero.
 *
 * @param[in] a      First operand
 * @param[in] b      Second operand
 * @param[in] acc_in Accumulator to add into, in units of 2**ACC_LSB
 * @return           The new accumulator value
 */
static int64_t model_mac(f16_t a, f16_t b, int64_t acc_in) {
    uint32_t sign_a = (uint32_t)a >> 15;
    uint32_t exp_a = ((uint32_t)a >> F16_FRAC_BITS) & 0x1Fu;
    uint32_t frac_a = (uint32_t)a & ((1u << F16_FRAC_BITS) - 1);

    uint32_t sign_b = (uint32_t)b >> 15;
    uint32_t exp_b = ((uint32_t)b >> F16_FRAC_BITS) & 0x1Fu;
    uint32_t frac_b = (uint32_t)b & ((1u << F16_FRAC_BITS) - 1);

    // a zero exponent covers both zero and subnormals, and both flush to zero,
    // contributing nothing to the accumulator
    if (exp_a == 0 || exp_b == 0) {
        return acc_in;
    }

    uint64_t mant_a = (uint64_t)(1u << F16_FRAC_BITS) | frac_a;
    uint64_t mant_b = (uint64_t)(1u << F16_FRAC_BITS) | frac_b;
    uint64_t mant_prod = mant_a * mant_b;

    // shift the product so its LSB lines up with the accumulator's; a product
    // lying entirely outside the accumulator's window contributes nothing
    int shift_amt = (int)exp_a + (int)exp_b - SHIFT_BIAS;

    uint64_t aligned;

    if (shift_amt <= -PROD_WIDTH || shift_amt >= ACC_WIDTH) {
        aligned = 0;
    } else if (shift_amt < 0) {
        aligned = mant_prod >> (unsigned)(-shift_amt);
    } else {
        aligned = (mant_prod << (unsigned)shift_amt) & ACC_MASK;
    }

    int64_t addend = (sign_a ^ sign_b) ? -(int64_t)aligned : (int64_t)aligned;

    return acc_trunc(acc_in + addend);
}

/**
 * @brief Converts a fixed-point accumulator to float16, mirroring fix_to_float.sv
 *
 * Rounds to nearest, ties to even. Values too large clamp to the largest finite
 * magnitude rather than to infinity, and values too small flush to zero.
 *
 * @param[in] fixed Accumulator value, in units of 2**ACC_LSB
 * @return          Equivalent float16 bit pattern
 */
static f16_t model_fix_to_f16(int64_t fixed) {
    uint32_t sign = (fixed < 0) ? 1u : 0u;
    f16_t sign_bit = (f16_t)(sign << 15);

    uint64_t mag = sign ? (uint64_t)(-fixed) : (uint64_t)fixed;

    if (mag == 0) {
        return sign_bit;
    }

    // index of the most significant set bit
    int msb_idx = 63 - __builtin_clzll(mag);

    // left-align the magnitude so its leading one lands in the top bit, putting
    // the fraction and the rounding bits at fixed positions
    uint64_t norm = mag << (MAG_WIDTH - 1 - msb_idx);

    int frac_lsb = MAG_WIDTH - 2 - F16_FRAC_BITS;  // 37

    uint32_t frac = (uint32_t)((norm >> (frac_lsb + 1)) & ((1u << F16_FRAC_BITS) - 1));
    uint32_t round_bit = (uint32_t)((norm >> frac_lsb) & 1u);
    uint32_t sticky = ((norm & (((uint64_t)1 << frac_lsb) - 1)) != 0) ? 1u : 0u;

    int exp_biased = msb_idx + EXP_OFFSET;

    uint32_t round_up = round_bit & (sticky | (frac & 1u));

    // a carry out means the fraction was all ones and has wrapped to all zeros,
    // which is the correct mantissa for the incremented exponent
    uint32_t frac_sum = frac + round_up;
    int exp_rounded = exp_biased + (int)(frac_sum >> F16_FRAC_BITS);

    // an exponent field of all ones is reserved, and a field of zero means zero
    // or subnormal
    if (exp_rounded <= 0) {
        return sign_bit;
    }

    if (exp_rounded >= F16_EXP_FIELD_MAX) {
        return (f16_t)(sign_bit | F16_MAX_BITS);
    }

    return (f16_t)(sign_bit
                   | ((uint32_t)exp_rounded << F16_FRAC_BITS)
                   | (frac_sum & ((1u << F16_FRAC_BITS) - 1)));
}

void device_model_block_matmul(const f16_t *a_block, const f16_t *b_block,
                               int blk_m, int blk_k, int blk_n,
                               f16_t *c_block) {
    const int D = FPGA_ARRAY_DIM;

    int tiles_i = blk_m / D;
    int tiles_j = blk_n / D;

    // loop across the block's output tiles
    for (int ti = 0; ti < tiles_i; ti++) {
        for (int tj = 0; tj < tiles_j; tj++) {
            f16_t *tile = c_block + ((size_t)ti * tiles_j + tj) * D * D;

            // loop across the tile's elements
            for (int r = 0; r < D; r++) {
                for (int c = 0; c < D; c++) {
                    int64_t acc = 0;

                    // The array zeroes the accumulators once per tile and runs
                    // every slice into them consecutively. Because each product
                    // is truncated independently before being added, the sum is
                    // order-independent and this plain k loop matches the
                    // array's slice-by-slice accumulation exactly.
                    for (int k = 0; k < blk_k; k++) {
                        acc = model_mac(a_block[(size_t)(ti * D + r) * blk_k + k],
                                        b_block[(size_t)k * blk_n + (tj * D + c)],
                                        acc);
                    }

                    tile[r * D + c] = model_fix_to_f16(acc);
                }
            }
        }
    }
}
