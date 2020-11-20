from math import sqrt

# a = 187. / 6400.
# b = 0.15 - 40. * a
# c = 400. * a - 2.

# a = 230./8000.
# b = (5.-400.*a)/20.
# c = 0

a = 113. / 2940.
b = - 241 / 147.
c = 900. * a

integer = 100.
y = round(a * integer**2 + b * integer + c)
print (str(y))

bi_range = 255.
y = round(-b / (2. * a) + sqrt(b ** 2 / a ** 2 / 4. - (c - bi_range) / a))
print (str(y))