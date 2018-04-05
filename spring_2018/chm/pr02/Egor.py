# coding=utf-8
# Задание 01
#
# Формат файла:
# n -- размерность матрицы
# Матрица [n, n], построчно
# Вектор [n], в одну строку

import numpy as np
import sys

output_precision = 3
eps = 1e-15


def read_matrix_file(n, m, f):
    matrix = np.zeros([n, m])
    for i in range(n):
        matrix[i] = np.fromstring(f.readline(), dtype=float, sep=' ')
    return matrix


def read_file(input_file):
    f = open(input_file, "r")
    n = int(f.readline())
    A = read_matrix_file(n, n, f)
    b = np.fromstring(f.readline(), dtype=float, sep=' ')
    return n, A, b


def LU(A, n, strict, swaps_restricted):
    A = A.copy()
    swaps = []
    L = np.zeros([n, n])
    U = np.zeros([n, n])
    for i in range(n):
        if abs(A[i][i]) < eps:
            if swaps_restricted:
                raise ValueError("Unable to swap rows because of restriction, interrupted.")
            fixed = False
            for j in range(i + 1, n):
                if abs(A[j][i]) >= eps:
                    A[[i, j]] = A[[j, i]]
                    fixed = True
                    swaps.append((i, j))
                    break
            if not fixed and strict:
                raise ValueError("det = 0, interrupted because of strict mode.")
        for j in range(i, n):
            U[i][j] = A[i][j]
            L[j][i] = A[j][i]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]
                L[j][i] -= L[j][k] * U[k][i]
            if strict and abs(U[i][i]) < eps:
                raise ValueError("det = 0, interrupted because of strict mode.")
            if abs(U[i][i]) >= eps:
                L[j][i] /= U[i][i]
    return L, U, swaps


def solve_U(U, b, n, strict):
    multipleSolutions = False
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - sum(U[i, i + 1:] * x[i + 1:])
        if abs(U[i][i]) > eps:
            x[i] /= U[i][i]
        else:
            if strict:
                raise ValueError("det = 0, interrupted because of strict mode.")
            if abs(x[i] - U[i][i]) > eps:
                raise ValueError("System has no solution, interrupted.")
            multipleSolutions = True
    return x, multipleSolutions


def solve_L(L, b, n, strict):
    multipleSolutions = False
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i] - sum(L[i, :i] * x[:i])
        if abs(L[i][i]) >= eps:
            x[i] /= L[i][i]
        else:
            if strict:
                raise ValueError("det = 0, interrupted because of strict mode.")
            if abs(x[i] - L[i][i]) >= eps:
                raise ValueError("System has no solution, interrupted.")
            multipleSolutions = True
    return x, multipleSolutions


def solve(A, b, n, strict, swaps_restricted):
    b = b.copy()
    L, U, swaps = LU(A, n, strict, swaps_restricted)
    for swap in swaps:
        b[[swap[0], swap[1]]] = b[[swap[1], swap[0]]]
    y, multipleSolutions_L = solve_L(L, b, n, strict)
    x, multipleSolutions_U = solve_U(U, y, n, strict)
    for swap in reversed(swaps):
        x[[swap[0], swap[1]]] = x[[swap[1], swap[0]]]
    if multipleSolutions_L or multipleSolutions_U:
        print("-----------------ALERT!-----------------")
        print("Solution isn't unique.")
        print("-----------------ALERT!-----------------\n")
    return x


def add_noise(v, n, delta=0.01):
    noise = (np.random.random_sample(n) * 2 - np.ones(n)) * delta
    noise *= v
    return v + noise


def calc_relative_change(v, old_v):
    return np.linalg.norm(v - old_v) / np.linalg.norm(old_v) * 100.


def main(input_file, full_info, strict, noise, swaps_restricted):
    np.set_printoptions(precision=output_precision)
    n, A, b = read_file(input_file)

    if full_info:
        print("-----------------Initial system-----------------\n")
        print("{}-dimensional system".format(n))
        print("A:\n{}\n".format(A))
        print("b:\n{}\n".format(b))
    try:
        x = solve(A, b, n, strict, swaps_restricted)
        print("-----------------Results-----------------\n")
        print("Found x = \n{}\n".format(x))
        if full_info:
            print("Ax - b = \n{}\n".format(np.dot(A, x) - b))
            print("relative error = {:.3f}%\n".format(calc_relative_change(np.dot(A, x), b)))

        if noise:
            try:
                b_noise = add_noise(b.copy(), n)
                x_noise = solve(A, b_noise, n, strict, swaps_restricted)
                print("-----------------Results with noise-----------------\n")
                print("b after adding noise = \n{}\n".format(b_noise))
                if full_info:
                    print("relative b change = {:.3f}%\n".format(calc_relative_change(b_noise, b)))
                print("x after adding noise = \n{}\n".format(x_noise))
                if full_info:
                    print("relative x change = {:.3f}%\n".format(calc_relative_change(x_noise, x)))

            except ValueError as error:
                print(error.message)
    except ValueError as error:
        print(error.message)


if __name__ == '__main__':
    filename = 'input.txt'
    full_mode = False
    strict_mode = False
    with_noise = False
    no_swaps = False

    arg_len = len(sys.argv)
    if arg_len >= 2:
        filename = sys.argv[1]
    for i in range(2, arg_len):
        if sys.argv[i] == '-full':
            full_mode = True
        elif sys.argv[i] == '-strict':
            strict_mode = True
        elif sys.argv[i] == '-noise':
            with_noise = True
        elif sys.argv[i] == '-no-swaps':
            no_swaps = True
        else:
            print("Unidentified parameter {}".format(sys.argv[i]))

    main(filename, full_mode, strict_mode, with_noise, no_swaps)
