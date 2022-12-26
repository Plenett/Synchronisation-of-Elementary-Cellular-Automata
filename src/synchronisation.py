from numpy import ndarray
from typing import Callable
from elementary_cellular_automata import ElementaryCellularAutomaton
import numpy as np


def synchronisation_random(CA: ElementaryCellularAutomaton, driver_initial: ndarray[int], replica_initial: ndarray[int],
                           p_coupling: float, time: int) -> ndarray[float]:
    """
    Compute the random pinching synchronisation
    :param CA: Rule of the Driver and the Replica
    :param driver_initial: Initial configuration of the driver
    :param replica_initial: Initial configuration of the replica
    :param p_coupling: Coupling probability
    :param time: Time Horizon for the synchronisation
    :return: Normalised Synchronisation Error
    """

    driver = np.asarray(driver_initial)
    replica = np.asarray(replica_initial)
    err = np.zeros(time)

    err[0] = np.mean(driver ^ replica)
    for t in range(1, time):
        # Step Systems
        driver = CA.step(driver)
        replica = CA.step(replica)

        # Measurements
        sensors = np.random.rand(len(driver)) < p_coupling
        replica[sensors] = driver[sensors]
        err[t] = np.mean(driver ^ replica)

    return err


def synchronisation_random_optimised(CA: ElementaryCellularAutomaton, driver_initial: ndarray[int],
                                     replica_initial: ndarray[int],
                                     p_coupling: float, time: int) -> ndarray[float]:
    """
        Compute the random pinching optimised synchronisation
        :param CA: Rule of the Driver and the Replica
        :param driver_initial: Initial configuration of the driver
        :param replica_initial: Initial configuration of the replica
        :param p_coupling: Coupling probability
        :param time: Time Horizon for the synchronisation
        :return: Normalised Synchronisation Error
        """