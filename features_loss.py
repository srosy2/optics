import torch
from rayoptics.environment import *
from rayoptics.parax.thirdorder import aspheric_seidel_contribution
from rayoptics.optical.model_constants import ht, slp, aoi

"""переделанная функция"""


def compute_third_order_cast(opt_model):
    seq_model = opt_model.seq_model
    n_before = seq_model.central_rndx(0)
    parax_data = opt_model['analysis_results']['parax_data']
    ax_ray, pr_ray, fod = parax_data
    opt_inv = fod.opt_inv
    opt_inv_sqr = opt_inv * opt_inv

    third_order = {}

    # Transfer from object
    p = 0
    pd_index = ['S-I', 'S-II', 'S-III', 'S-IV', 'S-V']
    for c in range(1, len(ax_ray) - 1):
        n_after = seq_model.central_rndx(c)
        n_after = n_after if seq_model.z_dir[c] > 0 else -n_after
        cv = seq_model.ifcs[c].profile_cv

        A = n_after * ax_ray[c][aoi]
        Abar = n_after * pr_ray[c][aoi]
        P = cv * (1. / n_after - 1. / n_before)
        delta_slp = ax_ray[c][slp] / n_after - ax_ray[p][slp] / n_before
        SIi = -A ** 2 * ax_ray[c][ht] * delta_slp
        SIIi = -A * Abar * ax_ray[c][ht] * delta_slp
        SIIIi = -Abar ** 2 * ax_ray[c][ht] * delta_slp
        SIVi = -opt_inv_sqr * P
        delta_n_sqr = 1. / n_after ** 2 - 1. / n_before ** 2
        SVi = -Abar * (Abar * Abar * delta_n_sqr * ax_ray[c][ht] -
                       (opt_inv + Abar * ax_ray[c][ht]) * pr_ray[c][ht] * P)

        scoef = pd.Series([SIi, SIIi, SIIIi, SIVi, SVi], index=pd_index)
        col = str(c)
        third_order[col] = scoef

        # handle case of aspheric profile
        if hasattr(seq_model.ifcs[c], 'profile'):
            to_asp = aspheric_seidel_contribution(seq_model, parax_data, c,
                                                  n_before, n_after)
            if to_asp:
                ascoef = pd.Series(to_asp, index=pd_index)
                third_order[col + '.asp'] = ascoef

        p = c
        n_before = n_after

    third_order_df = pd.DataFrame(third_order, index=pd_index)
    # third_order = third_order_df.sum(axis='columns')
    third_order = third_order_df.values.reshape(-1)
    return torch.tensor(list(third_order) + list(ax_ray.reshape(-1)) + list(pr_ray.reshape(-1)))
