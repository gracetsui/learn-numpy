import numpy as np

a = np.array([1,2,3,4,5])
print(a)
print(a[1])
print(a[1:4])
print(a[-1])
print(type(a))

a[2] = 10
print(a)

a_mul = np.array([[[1, 2, 3, 1],
                   [4, 5, 6, 1],
                   [7, 8, 9, 1]],
                  [[1, 1, 1, 1],
                   [1, 1, 1, 1],
                   [1, 1, 1, 1]]])

print(a_mul.shape)
print(a_mul.ndim)
print(a_mul.size)
print(a_mul.dtype)

print(a_mul[0])
print(a_mul[0, 1])

b = np.array([[1,2,3],
              [4,"5",6],
              [7,8,9]], dtype=np.float32)

print(b.dtype)
print(b[1][1].dtype)
print(b[1,1])

d = {'1': 'A'}

c = np.array([[1,2,3],
              [4,d,6],
              [7,8,"hello"]])

print(c.dtype)
print(type(c[1][0]))

e = np.array([[1,2,3],
              [4,d,6],
              [7,8,9]], dtype="<U7")

print(e.dtype)
print(type(e[1][0]))

# https://numpy.org/doc/stable/user/basics.types.html

###########################################################

f = np.full((2,3,4), 9)
print(f)

g = np.zeros((10,5,2))
print(g)

h = np.ones((10,5,2))
print(h)

i = np.empty((5,5,5))
print(i)

x_values = np.arange(0, 100, 5)
print(x_values)

x_values = np.arange(0, 1000, 5)
print(x_values)

x_values = np.arange(0, 1000, 1001)
print(x_values)

print(np.nan)
print(np.inf)

print(np.isnan(np.nan))
print(np.isinf(np.inf))

print(np.sqrt(-1))
print(np.array([10]) / 0)

###########################################################

l1 = [1,2,3,4,5]
l2 = [6,7,8,9,0]

a1 = np.array(l1)
a2 = np.array(l2)

print(l1 * 5)
print(a1 * 5)


print(l1 + l2)
print(a1 + a2)
print(a1 / a2)
print(a1 - a2)

b1 = np.array([1,2,3])
b2 = np.array([[1],
               [2]])

print(b1 + b2)

j = np.array([[1,2,3],
              [4,5,5]])


print(np.sqrt(j))
print(np.cos(j))
print(np.tan(j))

# https://numpy.org/doc/stable/reference/routines.math.html

###########################################################

k = np.array([1,2,3])

k = np.append(a, [7,8,9])
np.insert(a, 3, [4,5,6])

print(k)

print(np.delete(a, 1))
print(np.delete(a, 3))
print(np.delete(a, 4))

l = np.array([[1,2,3,4,5],
             [6,7,8,9,10],
             [11,12,13,14,15],
             [16,17,18,19,20]])

print(l.shape)
print(l.reshape((5,4)))
print(l.reshape((20,1)))
print(l.reshape((2, 2, 5)))

l.reshape(10, 2)
print(l)

l.resize((10,2))
print(l)

print(l.flatten())
print(l.ravel())

var1 = l.flatten
print(var1)
print(l)

m = np.array([[1,2,3,4,5],
             [6,7,8,9,10],
             [11,12,13,14,15],
             [16,17,18,19,20]])

var = [v for v in m.flat]
print(var)

print(m.transpose)
print(m.T)
print(m.swapaxes(0,1))

###########################################################

c1 = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])

c2 = np.array([[11,12,13,14,15],
               [16,17,18,19,20]])

n = np.concatenate((a1, a2), axis=0)
print(n)

n = np.stack((a1, a2), axis=1)
print(n)

n = np.hstack((a1, a2))
print(n)

n = np.vstack((a1, a2))
print(n)

o = np.array([[1,2,3,4,5],
              [6,7,8,9,10],
              [11,12,13,14,15],
              [16,17,18,19,20]])

print(np.split(o, 2, axis=0))
print(np.split(o, 4, axis=0))

p = np.array([[1,2,3,4,5,6],
              [7,8,9,10,11,12],
              [13,14,15,16,17,18],
              [19,20,21,22,23,24]])

#print(np.split(a, 2, axis=1))
#print(np.split(a, 3, axis=1))
#print(np.split(a, 6, axis=1))

print(p.min())
print(p.max())
print(p.mean())
print(p.std())
print(np.median(p))

###########################################################

numbers = np.random.randint(90, 100, size=(2,3,4))
print(numbers)

numbers2 = np.random.binomial(10, p=0.5, size=(5, 10))
print(numbers2)

numbers3 = np.random.normal(loc=170, scale=15, size=(5, 10))
print(numbers3)

numbers4 = np.random.choice([10,20,30,40,50], size=(5, 10))
print(numbers4)

q = np.array([[1,2,3,4,5,6],
              [7,8,9,10,11,12],
              [13,14,15,16,17,18],
              [18,19,20,21,22,23]])

np.save("myarray.py", q)

np.savetxt("myarray.csv", q, delimiter=",")

r = np.loadtxt("myarray.csv", delimiter=",")
print(r)
