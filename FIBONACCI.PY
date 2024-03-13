def fibonacci(n):
    fib_series = [0, 1]

    for i in range(2, n):
        next_num = fib_series[-1] + fib_series[-2]
        fib_series.append(next_num)

    return fib_series[:n]


n = int(input("Enter the number of Fibonacci numbers to generate: "))
print("Fibonacci numbers:", fibonacci(n))


