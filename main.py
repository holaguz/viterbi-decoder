#!/usr/bin/env python3

import random
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class TrellisNode:
    metric: int = 0xFFFF
    source_state: int = -1


def conv_encode_bytes(message: bytes, state=0) -> list[list[bool]]:
    """
    Encode a list of bytes. The state is initialized to zero. Returns a list of codewords.
    """
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
    out_1 = ((state & G1).bit_count() + input) & 1
    out_2 = ((state & G2).bit_count() + input) & 1
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

    dp = [[TrellisNode()] * num_states]  # [len(input), num_states]
    dp[0][0] = TrellisNode(0, -1)

    for sym in input:
        prev_col = dp[-1]
        new_col = [TrellisNode()] * num_states

        sym = bits_to_number(sym)
        for state, node in enumerate(prev_col):
            for t in [False, True]:
                next_state, word = conv_encode(t, state)
                updated_metric = node.metric + hamming_distance(
                    sym, bits_to_number(word)
                )

                if updated_metric < new_col[next_state].metric:
                    new_col[next_state] = TrellisNode(updated_metric, state)

        dp.append(new_col)

    # Discard the source state
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


# Returns True if the string was decoded succcesfully
def simulate(input_len: int = 32, num_errors: int = 0, verbose=False) -> bool:
    printv = print if verbose else lambda _: ()

    num_states = 64

    # Generate a random input.
    input = random.randbytes(input_len)

    # Encode the information bits.
    encoded = conv_encode_bytes(input)
    printv(f"{input.hex()} -> {encoded}")

    # Insert some errors
    errors = random.sample(range(len(encoded) * 2), num_errors)
    for error in errors:
        encoded[error // 2][error & 1] ^= True

    # Decode the codewords into information bits.
    dp = decode(encoded, num_states)
    decoded = []

    # Backtrack the Trellis.
    state = dp[-1].index(min(dp[-1], key=lambda x: x.metric))
    for step in dp[::-1]:
        # Because we insert the bits to the LFSR from the left,
        # we can check the MSB to determine the value of the bit
        # that caused the transition.
        decoded.append(state & 32 != 0)

        # The next best state is the state that brought us here
        state = step[state].source_state

    # Because we are backtracking, the order of decoded is inverted.
    decoded.reverse()

    # Parse the bits as bytes.
    bytes_ = bits_to_bytes(decoded)[::-1]

    printv("SRC: " + input.hex())
    printv("DEC: " + bytes_.hex())
    printv("      " + " ".join(["^" if i != j else " " for i, j in zip(bytes_, input)]))
    printv(f"{"Good" if bytes_ == input else "Bad"}")

    return bytes_ == input


if __name__ == "__main__":
    random.seed(42)
    test()

    niters = 128
    string_length = 32
    print("Input length:", string_length)
    for num_errors in range(string_length // 2):
        bad = 0
        for iters in range(niters):
            bad += 1 if not simulate(string_length, num_errors) else 0
        print(f"Errors: {num_errors}, Failures: {bad}/{niters} ({bad / niters:.2%})")
