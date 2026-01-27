from math import sqrt
import inspect
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


def exponential(t, y, alpha, beta):
    """Function modeling the ODE of the exponential growth model.

    :param t: Scalar time value at which differential equation is evaluated (unused but needed for interfacing)
    :param y: Scalar value of the model at the specified time
    :param alpha: Growth rate parameter
    :param beta: Death rate parameter
    :return: Value of the derivative of the model at the specified time
    """
    return (alpha - beta) * y


def logistic(t, y, gamma, kappa):
    """Function modeling the ODE of the logistic growth model.

    :param t: Scalar time value at which differential equation is evaluated (unused but needed for interfacing)
    :param y: Scalar value of the model at the specified time
    :param gamma: Net growth rate parameter
    :param kappa: Carrying capacity parameter
    :return: Value of the derivative of the model at the specified time
    """
    return gamma * y * (1 - (y / kappa))

def classic_gompertz(t, y, gamma, delta):
    """Function modeling the ODE of the Classic Gompertz growth model.

    :param t: Scalar time value at which differential equation is evaluated (unused but needed for interfacing)
    :param y: Scalar value of the model at the specified time
    :param gamma: Net growth rate parameter
    :param delta: Derived from net growth rate and carrying capacity
    :return: Value of the derivative of the model at the specified time
    """
    return y * (delta - gamma * np.log(y))

def general_gompertz(t, y, gamma, delta, lamda):
    """Function modeling the ODE of the Classic Gompertz growth model.

    :param t: Scalar time value at which differential equation is evaluated (unused but needed for interfacing)
    :param y: Scalar value of the model at the specified time
    :param gamma: Net growth rate parameter
    :param delta: Derived from net growth rate and carrying capacity
    :param lamda: Additional tuning parameter
    :return: Value of the derivative of the model at the specified time
    """
    return (y ** lamda) * (delta - (gamma * np.log(y)))

def classic_bertalanffy(t, y, alpha, beta):
    """Function modeling the ODE of the Classic Bertalanffy growth model.

    :param t: Scalar time value at which differential equation is evaluated (unused but needed for interfacing)
    :param y: Scalar value of the model at the specified time
    :param alpha: Growth rate parameter
    :param beta: Death rate parameter
    :return: Value of the derivative of the model at the specified time
    """
    return (alpha * (y ** 2 / 3)) - (beta * y)

def general_bertalanffy(t, y, alpha, beta, lamda):
    """Function modeling the ODE of the Classic Bertalanffy growth model.

    :param t: Scalar time value at which differential equation is evaluated (unused but needed for interfacing)
    :param y: Scalar value of the model at the specified time
    :param alpha: Growth rate parameter
    :param beta: Death rate parameter
    :param lamda: Additional tuning parameter
    :return: Value of the derivative of the model at the specified time
    """
    return alpha * (y ** lamda) - beta * y


class GrowthModel(ABC):
    """Abstract class representing the solution to a generic growth DE model.
    
    Model contains :meth:'solution' to return its values at an array of times, as well as methods to determine
    statistical metrics of its performance.

    :param base_ode: The base ODE of the model in function form
    """

    def __init__(self, base_ode):
        """Constructor method.
        """
        self.diff_eq = base_ode
        self.num_params = len(list(inspect.signature(self.diff_eq).parameters.values())[2:])

    @abstractmethod
    def solution(self, t_array, y0, *args):
        """Abstract method that returns model values at an array of times by numerically integrating the base DE model.

        :param t_array: Array of times at which model is evaluated
        :param y0: Initial condition used for numerical integration
        :param args: Additional parameters of the base DE model, must match what is expected by base DE model
        :return: Array of model values at the specified times
        """
        pass

    def sse(self, params, t_array, y_array):
        """Computes the sum of squared errors (SSE) for the model given DE parameters and a set of actual data

        :param params: Tuple of parameter values for the base DE model, must match what is expected by base DE model
        :param t_array: Array of time values at which data points were collected and model is evaluated
        :param y_array: Array of actual data points to which model predictions are compared at the specified times
        :return: Sum of squared errors for the model and parameters over the given data set
        """
        y_array_pred = self.solution(t_array, *params)
        return np.sum((y_array - y_array_pred) ** 2.0)

    def rmse(self, params, t_array, y_array):
        """Computes the root-mean-square error (RMSE) for the model given DE parameters and a set of actual data

        :param params: Tuple of parameter values for the base DE model, must match what is expected by base DE model
        :param t_array: Array of time values at which data points were collected and model is evaluated
        :param y_array: Array of actual data points to which model predictions are compared at the specified times
        :return: Root-mean-square error for the model and parameters over the given data set
        """
        y_array_pred = self.solution(t_array, *params)
        return sqrt(np.mean(np.sum((y_array - y_array_pred) ** 2.0)))

    def mae(self, params, t_array, y_array):
        """Computes the mean absolute error (MAE) for the model given DE parameters and a set of actual data

        :param params: Tuple of parameter values for the base DE model, must match what is expected by base DE model
        :param t_array: Array of time values at which data points were collected and model is evaluated
        :param y_array: Array of actual data points to which model predictions are compared at the specified times
        :return: Mean absolute error for the model and parameters over the given data set
        """
        y_array_pred = self.solution(t_array, *params)
        return np.mean(np.abs(y_array_pred - y_array))


class GrowthModelODE(GrowthModel):
    """Implemented subclass of :class:'GrowthModel' that represents the solution to an ordinary differential equation
     (ODE) growth model.
    """

    def solution(self, t_array, y0, *args):
        """Overrides :meth:'GrowthModel.solution' for a continuous model that can be numerically integrated over the 
        time interval in question using scipy.integrate.solve_ivp.
        
        When passed as a Callable to an optimization function, the initial condition 'y0' is also fit to the data.
        
        :param t_array: Array of times at which model is evaluated
        :param y0: Initial condition used for numerical integration
        :param args: Additional parameters of the base DE model, must match what is expected by base DE model
        :return: Array of model values at the specified times
        """
        sol = solve_ivp(self.diff_eq, [t_array[0], t_array[-1]], [y0], t_eval=t_array, args=args)
        return np.asarray(sol.y)[0]


class GrowthModelIDE(GrowthModel):
    """Implemented subclass of :class:'GrowthModel' that represents the solution to an impulsive differential equation 
    (IDE) growth model, which includes discontinuities as a result of periodic instantaneous scaling, and ODE modeling
    between discontinuities.
    
    :param impulse_period: Constant interval between subsequent impulses
    :param base_ode: The base ODE of the model in function form
    """

    def __init__(self, impulse_period, base_ode):
        """Constructor method.
        """
        self._treatment_period = impulse_period
        super().__init__(base_ode)
        self.num_params += 1

    def _impulse_event_factory(self, n_impulses):
        """Closure method that creates event functions to detect the next impulse given the number of impulses applied.

        :param n_impulses: Number of impulses that have been applied thus far during numerical integration of the IDE
        :return: Event function to detect the next impulse
        """
        def impulse_event(t, y):
            """Function given to scipy.integrate.solve_ivp to interrupt numerical integration when an impulse must occur

            :param t: Current time during numerical integration
            :param y: Current model value during numerical integration (unused but needed for interfacing)
            :return:
            """
            return t - (n_impulses * self._treatment_period + 1)

        # Give event function attributes that denote integration should cease and zero-crossing can be any direction
        impulse_event.terminal = True
        impulse_event.direction = 0
        return impulse_event

    def solution(self, t_array, y0, *args):
        """Overrides :meth:'GrowthModel.solution' for a discontinuous model containing impulsive scaling at regular
        intervals, and numerical integration of a base ODE between impulses using scipy.integrate.solve_ivp.

        When passed as a Callable to an optimization function, the initial condition 'y0' is also fit to the data.
        Additionally, the 'args' tuple contains one more parameter than the base ODE expects that acts as the
        impulsive scaling magnitude.

        :param t_array: Array of times at which model is evaluated
        :param y0: Initial condition used for numerical integration
        :param args: Additional parameters of the base DE model, including base ODE parameters and a scaling magnitude
        :return: Array of model values at the specified times
        """
        # Extract impulsive scaling magnitude 'phi' from 'args' tuple
        phi = args[0]

        # Initialize loop variables
        flag = True
        t0_n = t_array[0]
        y0_n = y0
        t_array_n = t_array
        y_array = []
        n = 0

        while flag:
            # Perform numerical integration between impulses
            sol = solve_ivp(self.diff_eq, [t0_n, t_array[-1]], [y0_n],
                            t_eval=t_array_n, events=self._impulse_event_factory(n), args=args[1:])

            # Determine start time of next numerical integration interval
            t0_n = np.asarray(sol.t)[-1]

            y_array_n = np.asarray(sol.y).tolist()
            if t0_n == t_array[-1]:
                # If next start time coincides with end of solution interval, flag final loop iteration
                flag = False
            else:
                # Else collect remaining times of interest and apply impulse to next model initial condition
                t_array_n = [t for t in t_array.tolist() if t >= t0_n]
                y0_n = y_array_n.pop()
                y0_n *= phi

            # Accumulate model values for latest interval between impulses and increment number of impulses applied
            y_array += y_array_n
            n += 1

        # Return array of model values at specified times
        return y_array
