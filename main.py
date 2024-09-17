#!/usr/bin/env python3

import random

# Convolutional codes are often characterized by the base code rate and the
# depth (or memory) of the encoder [n, k, K]. The base code rate is typically
# given as n / k, where n is the raw input data rate and k is the data rate of
# output channel encoded stream. The value K (or m) is the Constrain length of
# the encoder.
# This example is a [2, 1, 7] convolutional code.

G1 = 0o171  # Gen. 1
G2 = 0o133  # Gen. 2
C_LEN = (
    6  # Constrain length. Calculated as the sum of the length of all shift registers
)


def conv_encode_bytes(message: bytes, state=0) -> list[list[bool]]:
    """
    Encode a list of bytes. The state is initialized to zero. Returns a list of codewords.
    """
    state = 0
    result = []
    for b in message:
        for bit_idx in range(8):
            bit = b & (1 << bit_idx) > 0

            state, encoded = conv_encode(bit, state)
            result.append(encoded)
    return result


def conv_encode(input: bool, state=0) -> tuple[int, list[int]]:
    """
    Perform one iteration of encoding. Returns the updated state and the resulting codeword as a tuple.
    """

    # XXX: This can be made faster if we reverse the polynomials
    # and feed the LFSR from the right.
    out_1 = (state & G1).bit_count() % 2 ^ input
    out_2 = (state & G2).bit_count() % 2 ^ input
    state = ((state >> 1) | input << 5) & 63  # 6 bits

    return (state, [out_1, out_2])


def bits_to_number(bits: list):
    return sum([val << (len(bits) - idx - 1) for idx, val in enumerate(bits)])


def bits_to_bytes(bits: list):
    byte_list = [0] * (len(bits) // 8)
    for i in range(0, len(bits), 8):
        byte_list[i // 8] = bits_to_number(bits[i : i + 8])
    return bytes(byte_list)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def decode(input: list, num_states: int):
    assert len(input[0]) == 2  # Codewords should be 2 bits long
    state = 0

    dp = [[0xFFFF] * num_states]  # [len(input), num_states]
    dp[0][0] = 0

    for sym in input:
        prev_col = dp[-1]
        print(prev_col)
        new_col = [0xFFFF] * num_states

        sym = bits_to_number(sym)
        for state, metric in enumerate(prev_col):
            for t in [False, True]:
                next_state, word = conv_encode(t, state)
                new_col[next_state] = min(
                    new_col[next_state],
                    metric + hamming_distance(sym, bits_to_number(word)),
                )

        dp.append(new_col)

    # Discard the source state
    dp = dp[1:]
    return dp


def print_dp(dp):
    for i in range(len(dp[0])):
        print(f"{i:<3}", end="")
    print("")
    for c in dp:
        print(" ".join([f"{' ':^2}" if i == 0xFFFF else f"{i:^2}" for i in c]))


def test():
    assert 11 == bits_to_number([1, 0, 1, 1])
    assert hamming_distance(0b000, 0b111) == 3
    assert hamming_distance(0b100, 0b111) == 2
    assert hamming_distance(0b100, 0b110) == 1
    assert hamming_distance(1234, 1234) == 0


if __name__ == "__main__":
    test()
    random.seed(42)

    num_states = 64
    input = random.randbytes(32)

    # Input bits are read left to right
    encoded = conv_encode_bytes(input)
    print(f"{input.hex()} -> {encoded}")

    # Flip one bit
    encoded[1][0] ^= True

    dp = decode(encoded, num_states)  # [len(encoded), num_states]

    decoded = []
    best_metrics = []
    print_dp(dp)

    for i, dist in enumerate(dp[::-1]):
        state = [i for i in range(num_states)]
        tied = zip(state, dist)
        best = sorted(tied, key=lambda x: x[1])[0]
        decoded.append(best[0] & 32 != 0)
        best_metrics.append(best[1])

    bytes_ = bits_to_bytes(decoded)[::-1]

    print("SRC: " + input.hex())
    print("DEC: " + bytes_.hex())
    print(" " + " ".join(["^" if i != j else " " for i, j in zip(bytes_, input)]))
    print(best_metrics)

    print(f"{"Good" if bytes_ == input else "Bad"}")
