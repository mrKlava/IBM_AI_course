import numpy as np



list = [321, 312, 2, 213, 4, 643, 43, 32, 6, 3]
list = np.array(list)

msk = np.random.rand(len(list)) < 0.8

train = list[msk]
test = list[~msk]

print(train)
print(test)



# print(train)
# print(test)