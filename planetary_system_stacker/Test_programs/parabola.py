from math import sqrt

x0 = 0.
x1 = 50.
x2 = 200.

y0 = 0.
y1 = 13.
y2 = 255.

b = (y1 - y0) / (x1 - x0)
c = (y2 - y1 - b * (x2 - x1)) / (x1 ** 2 + x2 ** 2 - 2 * x1 * x2)
f = c * x1 ** 2 - b * x1 + y1
e = b - 2. * c * x1
g = -e / (2. * c)
h = g ** 2 - f / c

print("b = " + str(b))
print("c = " + str(c))
print("e = " + str(e))
print("f = " + str(f))
print("g = " + str(g))
print("h = " + str(h))

x = 200

print ("\nx: " + str(x))
if x < x1:
    y = y0 + b * (x - x0)
else:
    y = round(c * x ** 2 + e * x + f, 1)
print("y: " + str(y))

x = round(g + sqrt(h + y / c))
print ("x: " + str(x))
