#!/usr/bin/env python3

# Convolutional codes are often characterized by the base code rate and the
# depth (or memory) of the encoder [n, k, K]. The base code rate is typically
# given as n / k, where n is the raw input data rate and k is the data rate of
# output channel encoded stream. The value K (or m) is the Constrain length of
# the encoder.
# This example is a [2, 1, 7] convolutional code.

from typing import Optional


G1 = 0o171  # Gen. 1
G2 = 0o133  # Gen. 2
C_LEN = 7  # Constrain length. Calculated as the sum of the length of all shift registers

def conv_encode_bytes(message: bytes, state = 0):
    result = []
    for b in message:
        for bit_idx in range(8):
            bit = b & (1 << bit_idx) > 0

            state, encoded = conv_encode(bit, state)
            result.append(encoded[0])
            result.append(encoded[1])
    return result


def conv_encode(input: bool, state = 0):

    #XXX: Updating the state first might be wrong!
    state = ((state >> 1) | input << 5) & 63  # 6 bits
    out_1 = (state & G1).bit_count() % 2 ^ input
    out_2 = (state & G2).bit_count() % 2 ^ input

    return (state, [out_1, out_2])


def test():
    tcs = [
        [0xFF, 25919],
        [0x55, 31692],
        [0xAA, 7923],
    ]

    for tc in tcs:
        encoded = conv_encode_bytes(tc[0].to_bytes())
        numeric = sum(
            [val << (len(encoded) - pos - 1) for pos, val in enumerate(encoded)]
        )
        assert numeric == tc[1]

    assert 11 == bits_to_number([1, 0, 1, 1])

def bits_to_number(bits: list):
    return sum([val << (len(bits) - idx - 1) for idx, val in enumerate(bits)])

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

def build_state_tree() -> list:
    n_states = 2**C_LEN
    state_tree: list = [0] * 2 * n_states

    for state in range(n_states):
        for i in [False, True]:
            _, state_tree[2*state + i] = conv_encode(i, state)
    return state_tree

def decode(input: list, state_tree: list):
    state = 0
    depth = 8 # In number of input symbols

    dp: list[Optional[int]] = [None] * len(state_tree)
    dp[0] = 0

    get_child = lambda state, transition: state_tree[state + len(state_tree) // 2 + transition] # noqa

    # Get the next input
    sym = bits_to_number(input[:2])

    # Get the two possible next states
    s0, word0 = get_child(state, 0), conv_encode(False, state)[1]
    s1, word1 = get_child(state, 1), conv_encode(True, state)[1]

    print(f"Sym: {sym}")
    print(f"State {state} -> {word0}/0 -> {s0}")
    print(f"      {state} -> {word1}/1 -> {s1}")

    new_col = dp[-1]

    # Calculate the distance between the next state 


    # Append the new iteration

    pass

if __name__ == "__main__":
    test()

    state_tree = build_state_tree()
    print(state_tree)

    input = 0xDEADBABE.to_bytes(4)
    encoded = conv_encode_bytes(input)
    print(f"{input} -> {encoded}")

    rev_encoded = encoded[::-1]
    decoded = decode(rev_encoded, state_tree)

    # Try to decode the input

    # Flip a random bit
    # error = 1 << (69 % len(encoded))
    # encoded[error] ^= 1




