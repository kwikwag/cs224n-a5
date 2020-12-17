import math

def inv_sigma(y):
    # TODO : in certain conditions it might be better to calculate
    #        math.log(y/(y-1))
    return math.log(1/((1/y) - 1))

def inv_dy(dy, sign):
    # the solution in terms of y for dy = y(1-y)
    return (1/2) + sign * math.sqrt(1/4 - dy)

def inv_dsigma(dy, sign):
    # the solution in terms of y for dy = y(1-y)
    y = inv_dy(dy, sign)
    return inv_sigma(y)

def sigma(x):
    return 1/(1+math.exp(-x))