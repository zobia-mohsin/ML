import os


def clear(): return os.system("cls")  # cls=ckear screen


clear()  # call function to run


def remainder(num): return num % 2


print(remainder(5))

product = lambda x,y: x * y

print(product(2,3))

def testfunc(num):
    print(num)
    return lambda x: x * num

result10 = testfunc(10) #creating two functions here
result10 = lambda x: x * 10 #hardcoding 10 in both, 10 is num in both classes

result100 = testfunc(100)
result100 = lambda x: x * 100

#cannot do this, just calling the function here, GIVE IT AN ARGUMENT
#print(result10) 
print(result10(9))
print(result100(9))

def myfunc(n): 
    return lambda a: a * n

mydoubler = myfunc(2) #functions
mytripler = myfunc(3) #n is 3 and argument when printing is a

print(mydoubler(11))
print(mytripler(5))

numbers_list = [2,6,8,10,11,14,4,2,5,6]

#has two arguments, function and iterable. Iterable is numbers list will go through each of those numbers
filtered_list = list(filter(lambda num:(num>7), numbers_list)) #function lambda and numbers_list is the iterable
print(filtered_list)

#applies a function to all the elements of the list. two arguments. ONLY ONE EXPRESSION
mapped_list = list(map(lambda num: num %2, numbers_list))
print(mapped_list)

def addition(n):
    return n + n

numbers = [1,2,3,4]
result = map(addition, numbers)
print (list(result))

result = map(lambda n: n + n, numbers) #has to be same argument, all three n must match in lambda
print (list(result))