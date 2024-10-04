def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm