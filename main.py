from lab1.myFilter import my_filter

numbers = [1, 2, 3, 4, 5, 6]
even_numbers = my_filter(lambda x: x % 2 == 0, numbers)

print(list(even_numbers))  # [2, 4, 6]