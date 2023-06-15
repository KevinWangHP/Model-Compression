s = "hello"
s_list = list(s)
low = 0
high = len(s) - 1
yuan_list = ['a', 'e', 'i', 'o', 'u']
while low < high:
    while low < high and s_list[low] not in yuan_list:
        low += 1
    while low < high and s_list[high] not in yuan_list:
        high -= 1
    if low < high:
        s_list[low], s_list[high] = s_list[high], s_list[low]
        low += 1
        high -= 1
print("".join(s_list))