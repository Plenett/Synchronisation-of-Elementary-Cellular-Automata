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


def synchronisation_random_transition(CA: ElementaryCellularAutomaton,
                                      initial_configuration: Callable[[], tuple[ndarray[int], ndarray[int]]],
                                      p_coupling: ndarray[float], time: int, n: int) -> ndarray[ndarray[float]]:
    """
    Compute the random punching synchronisation transition diagram
    :param CA: Rule of the Driver and the Replica
    :param initial_configuration: Initial configuration for the driver and replica
    :param p_coupling: Coupling probabilities
    :param time: Time Horizon for the synchronisation
    :param n: Number of simulation for each probability
    :return: Array of size (n, len(p_coupling)) with the normalised synchronisation error at time t
    """

    err = np.zeros([n, len(p_coupling)])
    for i in range(n):
        for j in range(len(p_coupling)):
            (driver, replica) = initial_configuration()
            err[i, j] = synchronisation_random(CA, driver, replica, p_coupling[j], time)[-1]

    return err
