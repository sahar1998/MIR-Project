import os
from struct import pack
import sys
import part2, part1
import bitstring

persian_terms = []
positional_index_english = part2.file_read('positional_index_english.txt')
terms = list(positional_index_english.keys())


# print(terms)


# for term in part2.positional_index_persian:
#     persian_terms.append(term)
# for term in part2.positional_index_english:
#     terms.append(term)


def encode_number(number):
    bytes_list = []
    while True:
        bytes_list.insert(0, number % 128)
        if number < 128:
            break
        number = number // 128
    bytes_list[-1] += 128
    print(bytes_list)
    return pack('%dB' % len(bytes_list), *bytes_list)


def encode(numbers):
    bytes_list = []
    for number in numbers:
        bytes_list.append(encode_number(number))
    return b"".join(bytes_list)


def gamma_code(number):
    bin_number = '{0:08b}'.format(number)
    for i in range(len(bin_number)):
        if bin_number[i] == '1':
            bin_number = bin_number[i + 1:]
            break
    n = len(bin_number)
    result = ''
    for i in range(n):
        result += '1'
    result += '0' + bin_number
    int_result = [int(result, 2)]
    print(int_result)
    return pack('%dB' % len(int_result), *int_result)


def encode_gamma(numbers):
    bytes_list = []
    for number in numbers:
        bytes_list.append(gamma_code(number))
    return b"".join(bytes_list)


def prepare(postings):
    gaps_postings = {}
    for term in terms:
        gaps_postings[term] = [postings[term][0]]
        for i in range(1, len(postings[term])):
            k = list(postings[term][i].keys())
            values = list(postings[term][i].values())
            if i != 1:
                k_prev = list(postings[term][i - 1].keys())
                gaps_postings[term].append(k[0] - k_prev[0])
            else:
                gaps_postings[term].append(k[0])
            gaps_postings[term].append(len(values[0]))
            for j in range(len(values[0])):
                if j != 0:
                    gaps_postings[term].append(values[0][j] - values[0][j - 1])
                else:
                    gaps_postings[term].append(values[0][j])
    return gaps_postings


def vlb(postings):
    vlb_postings = {}
    for term in terms:
        vlb_postings[term] = encode(postings[term])
    return vlb_postings


def gamma(postings):
    gamma_postings = {}
    for term in terms:
        gamma_postings[term] = encode_gamma(postings[term])
    return gamma_postings


# part2.file_write('gamma_postings_en.txt', vlb(prepare(positional_index_english)))
# print(os.path.getsize('positional_index_english.txt'))
# print(os.path.getsize('vlb_postings_en.txt'))
