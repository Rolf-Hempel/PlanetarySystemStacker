from numpy import array, arange

from miscellaneous import Miscellaneous


def evaluate(coefficients, y, x):
    a_f, b_f, c_f, d_f, e_f, g_f = coefficients
    value = a_f * x ** 2 + b_f * y ** 2 + c_f * x * y + d_f * x + e_f * y + g_f
    print ("Function evaluation, y: " + str(y) + ", x: " +str(x) + ", value: " + str(value))
    return value

def create_function_values(coefficients):
    function_values = []
    for y in range(-1,2):
        for x in range(-1,2):
            function_values.append(evaluate(coefficients, y, x))
    return array(function_values)

def evaluate_around_optimum(coefficients, y_opt, x_opt):
    for y in arange(y_opt - 0.1, y_opt + 0.19, 0.1):
        for x in arange(x_opt - 0.1, x_opt + 0.19, 0.1):
            evaluate(coefficients, y, x)

def evaluate_derivatives(coefficients, y, x):
    a_f, b_f, c_f, d_f, e_f, g_f = coefficients
    df_dy = 2. * b_f * y + c_f * x + e_f
    df_dx = 2. * a_f * x + c_f * y + d_f
    print ("derivative_y: " + str(df_dy) + ", derivative_x: " + str(df_dx))


coeff = (1.5, 1.2, 2.5, -0.5, 0.7, 3.)
values = create_function_values(coeff)
print ("\nFunction values: " + str(values))

y, x = Miscellaneous.sub_pixel_solve(values)
evaluate(coeff, y, x)

y_old, x_old = Miscellaneous.sub_pixel_solve_old(values)
print ("Old solution, y: " + str(y_old) + ", x: " + str(x_old))

evaluate_around_optimum(coeff, y_old, x_old)

evaluate_derivatives(coeff, y_old, x_old)
