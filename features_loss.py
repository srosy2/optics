import numpy as np
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

    fod = parax_data.fod
    first_order_features = np.array([fod.efl, fod.ffl, fod.pp1, fod.bfl, fod.ppk, fod.fno, fod.img_dist, fod.img_ht,
                                     fod.exp_dist, fod.exp_radius, fod.img_na, fod.opt_inv])
    first_order_features = first_order_features / np.max(np.abs(first_order_features))
    third_order_df = pd.DataFrame(third_order, index=pd_index)
    third_order_df = third_order_df / np.max(np.abs(third_order_df))
    third_order_abs = abs(third_order_df).mean(axis='columns').values
    third_order_mean = third_order_df.mean(axis='columns').values
    ax_ray = np.array(ax_ray)
    ax_ray = ax_ray / np.max(np.abs(ax_ray))
    ax_ray_abs = np.mean(np.abs(ax_ray), axis=0)
    ax_ray_mean = np.mean(ax_ray, axis=0)
    pr_ray[0][0] /= 3500000000
    pr_ray /= np.max(np.abs(pr_ray))
    pr_ray_abs = np.mean(np.abs(pr_ray), axis=0)
    pr_ray_mean = np.mean(pr_ray, axis=0)
    # np.concatenate([third_order_sum, third_order_mean, ax_ray_sum, ax_ray_mean, pr_ray_sum, pr_ray_mean])
    # third_order = third_order_df.values.reshape(-1)
    # return torch.tensor(list(third_order) + list(np.array(ax_ray).reshape(-1)) + list(np.array(pr_ray).reshape(-1)))
    return torch.tensor(np.concatenate([third_order_abs, third_order_mean, ax_ray_abs, ax_ray_mean,
                                        pr_ray_abs, pr_ray_mean, first_order_features]))
