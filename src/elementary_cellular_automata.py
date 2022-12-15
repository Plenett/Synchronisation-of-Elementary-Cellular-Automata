from numpy import ndarray
from scipy import signal
import numpy as np


class ElementaryCellularAutomaton:
    """Class to represent Wolfram's elementary cellular automata to study CA synchronisation"""

    def __init__(self, rule: int, boundary: str):
        """
        Constructor for the class ElementaryCellularAutomaton
        :param rule: rule of the CA (between 0 and 255)
        :param boundary: boundary condition ("periodic", "null" or "reflexive")
        """
        self.rule = rule
        self.boundary = boundary

    def step(self, configuration: ndarray[int]) -> ndarray[int]:
        """
        Method that computes the next configuration
        :param configuration: configuration of the CA
        :return: next configuration of the CA
        """

        flt = [[1, 2, 4]]
        match self.boundary:
            case "periodic":
                result = signal.convolve2d([configuration], flt, boundary='wrap', mode='same')[0]
            case "null":
                result = signal.convolve2d([configuration], flt, boundary='fill', mode='same')[0]
            case "reflexive":
                result = signal.convolve2d([configuration], flt, boundary='symm', mode='same')[0]
            case _:
                raise "Unknown boundary condition"

        result = np.right_shift(self.rule, result)
        result = np.mod(result, 2)

        return result

    def compute(self, configuration: ndarray[int], time: int) -> ndarray[ndarray[int]]:
        """
        Method that computes the 'time' next configuration
        :param configuration: initial configuration of the CA
        :param time: time horizon
        :return: 'time' next configurations
        """

        confs = np.zeros([time, len(configuration)]).astype(int)
        confs[0] = np.asarray(configuration)
        for t in range(1, time):
            confs[t] = self.step(confs[t-1])

        return confs
