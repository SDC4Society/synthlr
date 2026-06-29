import math

def round_to_sigfigs(num, sigfigs):
    if num == 0:
        return 0
    return round(num, sigfigs - int(math.floor(math.log10(abs(num)))) - 1)

def compare_sigfigs(num1, num2, sigfigs=3):
    return round_to_sigfigs(num1, sigfigs) == round_to_sigfigs(num2, sigfigs)
