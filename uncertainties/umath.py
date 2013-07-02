'''
Mathematical operations that generalize many operations from the
standard math module so that they also work on numbers with
uncertainties.

Examples:

  from umath import sin
  
  # Manipulation of numbers with uncertainties:
  x = uncertainties.ufloat(3, 0.1)
  print sin(x)  # prints 0.141120008...+/-0.098999...

  # The umath functions also work on regular Python floats:
  print sin(3)  # prints 0.141120008...  This is a Python float.

Importing all the functions from this module into the global namespace
is possible.  This is encouraged when using a Python shell as a
calculator.  Example:

  import uncertainties
  from uncertainties.umath import *  # Imports tan(), etc.
  
  x = uncertainties.ufloat(3, 0.1)
  print tan(x)  # tan() is the uncertainties.umath.tan function

The numbers with uncertainties handled by this module are objects from
the uncertainties module, from either the Variable or the
AffineScalarFunc class.

(c) 2009-2013 by Eric O. LEBIGOT (EOL) <eric.lebigot@normalesup.org>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.'''

from __future__ import division  # Many analytical derivatives depend on this

# Standard modules
import math
import sys
import itertools
import functools
import inspect

# Local modules
import uncertainties

from uncertainties import __author__, to_affine_scalar, AffineScalarFunc

###############################################################################

# We wrap the functions from the math module so that they keep track of
# uncertainties by returning a AffineScalarFunc object.

# Some functions from the math module cannot be adapted in a standard
# way so to work with AffineScalarFunc objects (either as their result
# or as their arguments):

# (1) Some functions return a result of a type whose value and
# variations (uncertainties) cannot be represented by AffineScalarFunc
# (e.g., math.frexp, which returns a tuple).  The exception raised
# when not wrapping them with wrap() is more obvious than the
# one obtained when wrapping them (in fact, the wrapped functions
# attempts operations that are not supported, such as calculation a
# subtraction on a result of type tuple).

# (2) Some functions don't take continuous scalar arguments (which can
# be varied during differentiation): math.fsum, math.factorial...
# Such functions can either be:

# - wrapped in a special way.

# - excluded from standard wrapping by adding their name to
# no_std_wrapping

# Function that copies the relevant attributes from generalized
# functions from the math module:
wraps = functools.partial(functools.update_wrapper,
                          assigned=('__doc__', '__name__'))

def wrap_module(module, wrapped, wrapfun, fixed_derivatives=None):
    """Wrap a math module to become uncertainties aware.

    :param module: the module where new functions are to be inserted.
    :param wrapped: the wrapped math module.
    :param wrapfun: a wrapping function that takes two arguments:
                    the function and its derivative.
    :param fixed_derivatives: a name to derivative mapping used to
                              add/override fixed derivatives.
    """

    ########################################
    # Wrapping of math functions:

    # Fixed formulas for the derivatives of some functions from the math
    # module (some functions might not be present in all version of
    # Python).  Singular points are not taken into account.  The user
    # should never give "large" uncertainties: problems could only appear
    # if this assumption does not hold.

    # Functions not mentioned in _fixed_derivatives have their derivatives
    # calculated numerically.

    # Functions that have singularities (possibly at infinity) benefit
    # from analytical calculations (instead of the default numerical
    # calculation) because their derivatives generally change very fast.
    # Even slowly varying functions (e.g., abs()) yield more precise
    # results when differentiated analytically, because of the loss of
    # precision in numerical calculations.

    #def log_1arg_der(x):
    #    """
    #    Derivative of log(x) (1-argument form).
    #    """
    #    return 1/x

    def log_der0(*args):
        """
        Derivative of math.log() with respect to its first argument.

        Works whether 1 or 2 arguments are given.
        """
        if len(args) == 1:
            return 1/args[0]
        else:
            return 1/args[0]/wrapped.log(args[1])  # 2-argument form

        # The following version goes about as fast:

        ## A 'try' is used for the most common case because it is fast when no
        ## exception is raised:
        #try:
        #    return log_1arg_der(*args)  # Argument number check
        #except TypeError:
        #    return 1/args[0]/wrapped.log(args[1])  # 2-argument form

    erf_coef = 2/wrapped.sqrt(wrapped.pi)  # Optimization for erf()

    fixed_derivatives = {
        # In alphabetical order, here:
        'acos': [lambda x: -1/wrapped.sqrt(1-x**2)],
        'acosh': [lambda x: 1/wrapped.sqrt(x**2-1)],
        'asin': [lambda x: 1/wrapped.sqrt(1-x**2)],
        'asinh': [lambda x: 1/wrapped.sqrt(1+x**2)],
        'atan': [lambda x: 1/(1+x**2)],
        'atan2': [lambda y, x: x/(x**2+y**2),  # Correct for x == 0
                  lambda y, x: -y/(x**2+y**2)],  # Correct for x == 0
        'atanh': [lambda x: 1/(1-x**2)],
        'ceil': [lambda x: 0],
        'copysign': [lambda x, y: (1 if x >= 0 else -1) * wrapped.copysign(1, y),
                     lambda x, y: 0],
        'cos': [lambda x: -wrapped.sin(x)],
        'cosh': [wrapped.sinh],
        'degrees': [lambda x: wrapped.degrees(1)],
        'erf': [lambda x: wrapped.exp(-x**2)*erf_coef],
        'erfc': [lambda x: -wrapped.exp(-x**2)*erf_coef],
        'exp': [wrapped.exp],
        'expm1': [wrapped.exp],
        'fabs': [lambda x: 1 if x >= 0 else -1],
        'floor': [lambda x: 0],
        'hypot': [lambda x, y: x/wrapped.hypot(x, y),
                  lambda x, y: y/wrapped.hypot(x, y)],
        'log': [log_der0,
                lambda x, y: -wrapped.log(x, y)/y/wrapped.log(y)],
        'log10': [lambda x: 1/x/wrapped.log(10)],
        'log1p': [lambda x: 1/(1+x)],
        'pow': [lambda x, y:
                    0. if y == 0
                    else y*wrapped.pow(x, y-1)  if x != 0 or y % 1 == 0
                    else float('nan'),
                lambda x, y:
                    0. if (x == 0) and (y > 0)
                    else wrapped.log(x) * wrapped.pow(x, y)],
        'radians': [lambda x: wrapped.radians(1)],
        'sin': [wrapped.cos],
        'sinh': [wrapped.cosh],
        'sqrt': [lambda x: 0.5/wrapped.sqrt(x)],
        'tan': [lambda x: 1+wrapped.tan(x)**2],
        'tanh': [lambda x: 1-wrapped.tanh(x)**2]
        }

    if fixed_derivatives:
        fixed_derivatives.update(fixed_derivatives)

    # Functions with numerical derivatives:
    num_deriv_funcs = ['fmod', 'gamma', 'isinf', 'isnan', 'lgamma', 'trunc']

    exported = []

    # Many built-in functions in the math module are wrapped with a
    # version which is uncertainty aware:

    # for (name, attr) in vars(math).items():
    for name in dir(wrapped):

        if name in fixed_derivatives:  # Priority to functions in fixed_derivatives
            derivatives = fixed_derivatives[name]
        elif name in num_deriv_funcs:
            # Functions whose derivatives are calculated numerically by
            # this module fall here (isinf, fmod,...):
            derivatives = []  # Means: numerical calculation required
        else:
            continue  # 'name' not wrapped by this module (__doc__, e, etc.)

        # Errors during the calculation of the derivatives are converted
        # to a NaN result: it is assumed that a mathematical calculation
        # that cannot be calculated indicates a non-defined derivative
        # (the derivatives in fixed_derivatives must be written this way):
        derivatives = map(uncertainties.nan_if_exception, derivatives)

        func = getattr(wrapped, name)

        setattr(module, name, wraps(wrapfun(func, derivatives), func))

        exported.append(name)

    ###############################################################################

    ########################################
    # Special cases: some of the functions from no_std_wrapping:

    ##########
    # The math.factorial function is not converted to an uncertainty-aware
    # function, because it does not handle non-integer arguments: it does
    # not make sense to give it an argument with a numerical error
    # (whereas this would be relevant for the gamma function).

    ##########

    # fsum takes a single argument, which cannot be differentiated.
    # However, each of the arguments inside this single list can
    # be a variable.  We handle this in a specific way:

    if sys.version_info[:2] >= (2, 6):

        # For drop-in compatibility with the math module:
        setattr(module, 'factorial', wrapped.factorial)
        exported.append('factorial')


        # We wrap math.fsum

        original_func = wrapped.fsum  # For optimization purposes

        # The function below exists so that temporary variables do not
        # pollute the module namespace:
        def wrapped_fsum():
            """
            Returns an uncertainty-aware version of math.fsum, which must
            be contained in _original_func.
            """

            # The fsum function is flattened, in order to use the
            # wrap() wrapper:

            flat_fsum = lambda *args: original_func(args)

            flat_fsum_wrap = wrapfun(
                flat_fsum, itertools.repeat(lambda *args: 1))

            return wraps(lambda arg_list: flat_fsum_wrap(*arg_list),
                         original_func)

        setattr(module, 'fsum', wrapped_fsum())
        exported.append('fsum')

    return exported

this_module = sys.modules[__name__]

##########
@uncertainties.set_doc(math.modf.__doc__)
def modf(x):
    """
    Version of modf that works for numbers with uncertainty, and also
    for regular numbers.
    """

    # The code below is inspired by uncertainties.wrap().  It is
    # simpler because only 1 argument is given, and there is no
    # delegation to other functions involved (as for __mul__, etc.).

    aff_func = to_affine_scalar(x)

    (frac_part, int_part) = math.modf(aff_func.nominal_value)

    if aff_func.derivatives:
        # The derivative of the fractional part is simply 1: the
        # derivatives of modf(x)[0] are the derivatives of x:
        return (AffineScalarFunc(frac_part, aff_func.derivatives), int_part)
    else:
        # This function was not called with an AffineScalarFunc
        # argument: there is no need to return numbers with uncertainties:
        return (frac_part, int_part)

@uncertainties.set_doc(math.ldexp.__doc__)
def ldexp(x, y):
    # The code below is inspired by uncertainties.wrap().  It is
    # simpler because only 1 argument is given, and there is no
    # delegation to other functions involved (as for __mul__, etc.).

    # Another approach would be to add an additional argument to
    # uncertainties.wrap() so that some arguments are automatically
    # considered as constants.

    aff_func = to_affine_scalar(x)  # y must be an integer, for math.ldexp

    if aff_func.derivatives:
        factor = 2**y
        return AffineScalarFunc(
            math.ldexp(aff_func.nominal_value, y),
            # Chain rule:
            dict((var, factor*deriv)
                 for (var, deriv) in aff_func.derivatives.iteritems()))
    else:
        # This function was not called with an AffineScalarFunc
        # argument: there is no need to return numbers with uncertainties:

        # aff_func.nominal_value is not passed instead of x, because
        # we do not have to care about the type of the return value of
        # math.ldexp, this way (aff_func.nominal_value might be the
        # value of x coerced to a difference type [int->float, for
        # instance]):
        return math.ldexp(x, y)

@uncertainties.set_doc(math.frexp.__doc__)
def frexp(x):
    """
    Version of frexp that works for numbers with uncertainty, and also
    for regular numbers.
    """

    # The code below is inspired by uncertainties.wrap().  It is
    # simpler because only 1 argument is given, and there is no
    # delegation to other functions involved (as for __mul__, etc.).

    aff_func = to_affine_scalar(x)

    if aff_func.derivatives:
        result = math.frexp(aff_func.nominal_value)
        # With frexp(x) = (m, e), dm/dx = 1/(2**e):
        factor = 1/(2**result[1])
        return (
            AffineScalarFunc(
                result[0],
                # Chain rule:
                dict((var, factor*deriv)
                     for (var, deriv) in aff_func.derivatives.iteritems())),
            # The exponent is an integer and is supposed to be
            # continuous (small errors):
            result[1])
    else:
        # This function was not called with an AffineScalarFunc
        # argument: there is no need to return numbers with uncertainties:
        return math.frexp(x)

###############################################################################
# Exported functions:

__all__ = wrap_module(this_module, math, uncertainties.wrap) + \
          ['modf', 'ldexp', 'frexp']


