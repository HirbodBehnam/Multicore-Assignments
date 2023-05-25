import random
size = int(input("Enter size of graph: "))
with open("test.txt", "w") as output:
    output.write(f"{size}\n")
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            output.write(f"a {i + 1} {j + 1} {random.randint(1, 1000)}\n")