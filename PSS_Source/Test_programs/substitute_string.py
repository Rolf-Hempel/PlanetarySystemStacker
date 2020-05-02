from miscellaneous import Miscellaneous

stack_f = 7
stack_p = None
box_size = 48
num_points = 401

string = Miscellaneous.compose_suffix(stack_f=stack_f, stack_p=stack_p, box_size=box_size,
                                      num_points=num_points)

print ("length of string: " + str(len(string)) + ", content: " + string)
