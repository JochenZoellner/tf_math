import numpy as np


def phi_array(delta_phi=0.01, phi_min=0.0, phi_max=np.pi, dtype=None):
    """ phi array start at 0 with possible pi as max, depends on delta_phi"""
    return np.arange(0.0, np.pi, delta_phi, dtype=dtype)


def phi_array_open(delta_phi=0.01, phi_min=0.0, phi_max=np.pi, dtype=None):
    """"phi array with open interval (0, pi)"""
    return np.arange(delta_phi, np.pi - delta_phi, delta_phi, dtype=dtype)


def phi_array_open_symetric_no90(delta_phi=0.01, phi_min=0.0, phi_max=np.pi, dtype=None):
    """exact delta_phi distance from 0 and pi and potentially a bit more to pi/2"""
    first_part = np.arange(delta_phi, np.pi / 2.0 - delta_phi, delta_phi, dtype=dtype)
    second_part = np.pi - np.flip(first_part)
    return np.concatenate((first_part, second_part))


def phi_array_open_no90(delta_phi=0.01, phi_min=0.0, phi_max=np.pi, dtype=None):
    """exact detla phi distance from 0 and pi/2 (upper part) and potentially a bit mor to pi/2 (left) and pi"""
    first_part = np.arange(delta_phi, np.pi / 2.0 - delta_phi, delta_phi, dtype=dtype)
    second_part = np.arange(np.pi / 2.0 + delta_phi, np.pi - delta_phi, delta_phi, dtype=dtype)
    return np.concatenate((first_part, second_part))


def phi_array_2pi_open_symetric_no90(delta_phi=0.01, phi_min=0.0, phi_max=np.pi, dtype=None):
    """exact delta_phi distance from 0 and 2pi and potentially a bit more to pi/2 and 3pi/2"""
    half_range = phi_array_2pi_open_symetric_no90(delta_phi, phi_min, phi_max, dtype)
    return np.concatenate((half_range, half_range + np.pi))


if __name__ == "__main__":
    print(phi_array_open_symetric_no90(), phi_array_open_symetric_no90().shape)
