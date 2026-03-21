def compute_SI(z, params):
    return params["SI_base"] * (1.0 + params["alpha"] * z)