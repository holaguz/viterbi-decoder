#!/usr/bin/env python3

# Convolutional codes are often characterized by the base code rate and the
# depth (or memory) of the encoder [n, k, K]. The base code rate is typically
# given as n / k, where n is the raw input data rate and k is the data rate of
# output channel encoded stream. The value K (or m) is the Constrain length of
# the encoder.
# This example is a [2, 1, 7] convolutional code.

import random

G1 = 0o171  # Gen. 1
G2 = 0o133  # Gen. 2
C_LEN = (
    6  # Constrain length. Calculated as the sum of the length of all shift registers
)


def conv_encode_bytes(message: bytes, state=0) -> list[bool]:
    result = []
    for b in message:
        for bit_idx in range(8):
            bit = b & (1 << bit_idx) > 0

            state, encoded = conv_encode(bit, state)
            result.append(encoded[0])
            result.append(encoded[1])
    return result


def conv_encode(input: bool, state=0):
    # XXX: Updating the state first might be wrong!
    out_1 = (state & G1).bit_count() % 2 ^ input
    out_2 = (state & G2).bit_count() % 2 ^ input
    state = ((state >> 1) | input << 5) & 63  # 6 bits

    return (state, [out_1, out_2])


def test():
    # tcs = [
    #     [0xFF, 25919],
    #     [0x55, 31692],
    #     [0xAA, 7923],
    # ]

    # for tc in tcs:
    #     encoded = conv_encode_bytes(tc[0].to_bytes())
    #     numeric = sum(
    #         [val << (len(encoded) - pos - 1) for pos, val in enumerate(encoded)]
    #     )
    #     assert numeric == tc[1]

    assert 11 == bits_to_number([1, 0, 1, 1])

    state_tree = build_state_tree()
    assert get_child(0, state_tree, False) == 0
    assert get_child(0, state_tree, True) == 32

    assert get_child(33, state_tree, False) == 16
    assert get_child(33, state_tree, True) == 48


def bits_to_number(bits: list):
    return sum([val << (len(bits) - idx - 1) for idx, val in enumerate(bits)])

def bits_to_bytes(bits: list):
    byte_list = [0] * (len(bits) // 8)
    for i in range(0, len(bits), 8):
        byte_list[i // 8] = bits_to_number(bits[i:i+8])
    return bytes(byte_list)

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_state_tree() -> list:
    n_states = 2**C_LEN
    state_tree: list = [0] * 2 * (n_states)

    for state in range(n_states):
        for i in [False, True]:
            out_state, transition = conv_encode(i, state)
            # print(f"{state} ({2*state+i}) -> {i*1} -> {out_state}")
            state_tree[2 * state + i] = out_state
    return state_tree


# Returns the next state based on the current state and the transition
def get_child(state: int, state_tree, transition: bool) -> int:
    return state_tree[2 * state + 1 * transition]


def decode(input: list, state_tree: list):
    assert len(input) % 2 == 0
    num_states = len(state_tree) // 2
    state = 0

    # [state, metric]
    dp = [[1 << 31 * num_states]]  # [len(input)//2, num_states]
    dp[0][0] = 0

    # Group the input in twos
    input = [input[i : i + 2] for i in range(0, len(input), 2)]

    for sym in input:
        prev_col = dp[-1]
        new_col = [1 << 31] * num_states

        sym = bits_to_number(sym)
        # print(f"Parsing {sym}")
        for state, metric in enumerate(prev_col):
            for t in [0, 1]:
                next_state, word = conv_encode(t == 1, state)
                new_col[next_state] = min(
                    new_col[next_state],
                    metric + hamming_distance(sym, bits_to_number(word)),
                )

        dp.append(new_col)

    dp = dp[1:]

    print(f"Returning trellis with depth {len(dp)}")
    return dp


if __name__ == "__main__":
    test()

    state_tree = build_state_tree()

    input = b'\xde\xad\xbe\xef'

    # Input bits are read left to right
    encoded = conv_encode_bytes(input)
    print(f"{input} -> {encoded}")

    num_errors = 1
    for i in random.sample(range(len(encoded)), num_errors):
        encoded[i] ^= 1

    dp = decode(encoded, state_tree) # [len(encoded), num_states]
    decoded = []

    for metrics in dp[::-1]:
        sm = zip(range(len(encoded)), metrics)
        state, metric = sorted(sm, key=lambda x: x[1])[0]
        decoded.append(state & 32 == 32)
        print(state, metric)

    # Endian is reversed, I don't know why
    print(bits_to_bytes(decoded))


