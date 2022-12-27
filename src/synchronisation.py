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
        if err[t - 1] != 0:
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

    driver = np.asarray(driver_initial)
    replica = np.asarray(replica_initial)
    initial_error = np.ones(len(replica)).astype(int)
    err = np.zeros(time)

    err[0] = np.mean(driver ^ replica)
    for t in range(1, time):
        # Step Systems
        driver = CA.step(driver)
        replica = CA.step(replica)

        # Measurements
        if err[t - 1] != 0:
            sensors_position = np.zeros(len(initial_error)).astype(int)
            for pos in np.where(initial_error == 1)[0]:
                sensors_position = sensors_position | propagate_error(pos, t, len(initial_error))

            sensors = sensors_position & (np.random.rand(len(driver)) * np.mean(sensors_position) < p_coupling).astype(int)
            measured_error = (driver ^ replica) & sensors
            replica[sensors.astype(bool)] = driver[sensors.astype(bool)]
            err[t] = np.mean(driver ^ replica)

            # Retropropagate Error
            for pos in np.where(measured_error == 1)[0]:
                initial_error = initial_error & propagate_error(pos, t, len(initial_error))

    return err


def propagate_error(position: int, time: int, size: int, alpha: float = 2.0, boundary: str = "periodic") -> ndarray[
    int]:
    """
    Compute the theoretical propagation of error given a position and a time horizon
    :param position: position of the initial error to propagate
    :param time: time horizon of the propagation
    :param size: size of the lattice
    :param alpha: error spread ratio of the error
    :param boundary: boundary condition of the lattice
    :return: lattice with the error spread
    """
    min_pos = position - np.ceil(time * alpha / 2.).astype(int)
    max_pos = position + np.ceil(time * alpha / 2.).astype(int) + 1

    lattice = np.zeros(size).astype(int)
    match boundary:
        case "periodic":
            if min_pos < 0:
                lattice[max(size + min_pos, 0):size] = 1
            if max_pos > size:
                lattice[0:min(max_pos - size, size)] = 1
            lattice[max(min_pos, 0):min(max_pos, size)] = 1

        case "null" | "reflexive":
            lattice[max(min_pos, 0):min(max_pos, size)] = 1

        case _:
            raise "Unknown boundary condition"

    return lattice
