from numpy import ndarray
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
    err = np.zeros(time).astype(int)
    err[0] = np.mean(np.abs(driver ^ replica))
    for t in range(1, time):
        driver = CA.step(driver)
        replica = CA.step(replica)

        sensors = np.random.rand(len(driver)) < p_coupling
        replica[sensors] = driver[sensors]

        err[t] = np.mean(np.abs(driver ^ replica))

    return err
