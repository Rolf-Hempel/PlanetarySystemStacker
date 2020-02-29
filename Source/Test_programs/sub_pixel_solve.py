from numpy import array, matmul, arange
from numpy.linalg import solve


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

def sub_pixel_solve_old(function_values):
    """
    Compute the sub-pixel correction for method "search_local_match".

    :param function_values: Matching differences at (3 x 3) pixels around the minimum found
    :return: Corrections in y and x to the center position for local minimum
    """

    # If the functions are not yet reduced to 1D, do it now.
    function_values_1d = function_values.reshape((9,))

    # There are nine equations for six unknowns. Use normal equations to solve for optimum.
    a_transpose = array(
        [[1., 0., 1., 1., 0., 1., 1., 0., 1.], [1., 1., 1., 0., 0., 0., 1., 1., 1.],
         [1., -0., -1., -0., 0., 0., -1., 0., 1.], [-1., 0., 1., -1., 0., 1., -1., 0., 1.],
         [-1., -1., -1., 0., 0., 0., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    a_transpose_a = array(
        [[6., 4., 0., 0., 0., 6.], [4., 6., 0., 0., 0., 6.], [0., 0., 4., 0., 0., 0.],
         [0., 0., 0., 6., 0., 0.], [0., 0., 0., 0., 6., 0.], [6., 6., 0., 0., 0., 9.]])

    # Right hand side is "a transposed times input vector".
    rhs = matmul(a_transpose, function_values_1d)

    # Solve for parameters of the fitting function
    # f = a_f * x ** 2 + b_f * y ** 2 + c_f * x * y + d_f * x + e_f * y + g_f
    a_f, b_f, c_f, d_f, e_f, g_f = solve(a_transpose_a, rhs)
    print ("\nSolve old, coeffs: " + str((a_f, b_f, c_f, d_f, e_f, g_f)))

    # The corrected pixel values of the minimum result from setting the first derivatives of
    # the fitting funtion in y and x direction to zero, and solving for y and x.
    denominator_y = c_f ** 2 - 4. * a_f * b_f
    if abs(denominator_y) > 1.e-10 and abs(a_f) > 1.e-10:
        y_correction = (2. * a_f * e_f - c_f * d_f) / denominator_y
        x_correction = (- c_f * y_correction - d_f) / (2. * a_f)
    elif abs(denominator_y) > 1.e-10 and abs(c_f) > 1.e-10:
        y_correction = (2. * a_f * e_f - c_f * d_f) / denominator_y
        x_correction = (-2. * b_f * y_correction - e_f) / c_f
    else:
        raise Exception("Sub-pixel shift cannot be computed, set to zero")

    return y_correction, x_correction

def sub_pixel_solve(function_values):
    """
    Compute the sub-pixel correction for method "search_local_match".

    :param function_values: Matching differences at (3 x 3) pixels around the minimum found
    :return: Corrections in y and x to the center position for local minimum
    """

    # If the functions are not yet reduced to 1D, do it now.
    function_values_1d = function_values.reshape((9,))

    # Solve the normal equations. Reduce the problem to a matrix multiplication with the matrix
    # "inv(a_transpose * a) * a_transpose" and the function value vector.
    m = [[0.16666667, -0.33333333, 0.16666667, 0.16666667, -0.33333333, 0.16666667, 0.16666667,
          -0.33333333, 0.16666667],
         [0.16666667, 0.16666667, 0.16666667, -0.33333333, -0.33333333, -0.33333333, 0.16666667,
          0.16666667, 0.16666667],
         [0.25, 0., -0.25, 0., 0., 0., -0.25, 0., 0.25],
         [-0.16666667, 0., 0.16666667, -0.16666667, 0., 0.16666667, -0.16666667, 0., 0.16666667],
         [-0.16666667, -0.16666667, -0.16666667, 0., 0., 0., 0.16666667, 0.16666667, 0.16666667],
         [-0.11111111, 0.22222222, -0.11111111, 0.22222222, 0.55555556, 0.22222222, -0.11111111,
          0.22222222, -0.11111111]]

    # Solve for parameters of the fitting function
    # f = a_f * x ** 2 + b_f * y ** 2 + c_f * x * y + d_f * x + e_f * y + g_f
    a_f, b_f, c_f, d_f, e_f, g_f = matmul(m, function_values_1d)
    print("\nSolve, coeffs: " + str((a_f, b_f, c_f, d_f, e_f, g_f)))

    # The corrected pixel values of the minimum result from setting the first derivatives of
    # the fitting funtion in y and x direction to zero, and solving for y and x.
    denominator_y = c_f ** 2 - 4. * a_f * b_f
    if abs(denominator_y) > 1.e-10 and abs(a_f) > 1.e-10:
        y_correction = (2. * a_f * e_f - c_f * d_f) / denominator_y
        x_correction = (- c_f * y_correction - d_f) / (2. * a_f)
    elif abs(denominator_y) > 1.e-10 and abs(c_f) > 1.e-10:
        y_correction = (2. * a_f * e_f - c_f * d_f) / denominator_y
        x_correction = (-2. * b_f * y_correction - e_f) / c_f
    else:
        raise Exception("Sub-pixel shift cannot be computed, set to zero")

    return y_correction, x_correction


coeff = (1.5, 1.2, 2.5, -0.5, 0.7, 3.)
values = create_function_values(coeff)
print ("\nFunction values: " + str(values))

y, x = sub_pixel_solve(values)
evaluate(coeff, y, x)

y_old, x_old = sub_pixel_solve_old(values)
print ("Old solution, y: " + str(y_old) + ", x: " + str(x_old))

evaluate_around_optimum(coeff, y_old, x_old)

evaluate_derivatives(coeff, y_old, x_old)
