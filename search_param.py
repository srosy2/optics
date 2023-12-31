import torch
from sklearn.model_selection import ParameterSampler
import numpy as np
from scipy.stats import uniform
from tqdm import tqdm
from check_os import Run
from scipy.optimize import minimize
from optic_model import calc_n, calc_abbe


def empty_start_random_search(quantity):
    d_coef = 0.02
    d_c = 0.5
    d_t = 7
    d_t_m = 3
    k = 1
    rng = np.random.RandomState(1)
    random_uniform_scale = list()
    for x in range(quantity):
        random_uniform_scale.append(uniform(loc=-d_c, scale=2 * d_c))
        random_uniform_scale.append(uniform(loc=0, scale=d_t))
        random_uniform_scale.append(uniform(loc=0, scale=k))
        random_uniform_scale.append(uniform(loc=-d_c, scale=2 * d_c))
        random_uniform_scale.append(uniform(loc=0, scale=d_t))
    for x in range(16 * quantity):
        random_uniform_scale.append(uniform(loc=-d_coef, scale=2 * d_coef))
    random_uniform_scale.append(uniform(loc=0, scale=d_t_m))
    for _ in tqdm(range(10000)):
        values = [uni.rvs(random_state=rng) for uni in random_uniform_scale]
        free_space = 7 - values[-1]
        sum_space = 0
        for x in range(quantity):
            sum_space += values[x * 5 + 1]
            sum_space += values[x * 5 + 4]
        if sum_space > free_space:
            norm = free_space / sum_space
            for x in range(quantity):
                values[x * 5 + 1] *= norm
                values[x * 5 + 4] *= norm
        delete = 0
        for x in range(quantity):
            if values[x * 5 + 1] < 0.1:
                delete += 0.1 - values[x * 5 + 1]
                values[x * 5 + 1] = 0.1

        for x in range(quantity):
            if values[x * 5 + 4] > delete / quantity:
                values[x * 5 + 4] -= delete / quantity
        param_values = list()
        for x in range(quantity):
            for i in range(5):
                if i == 2:
                    param_values.append(calc_n(values[x * 5 + i]))
                    param_values.append(calc_abbe(values[x * 5 + i]))
                else:
                    param_values.append(values[x * 5 + i])
        param_values.extend(values[quantity * 5: -1])
        main(np.array(param_values))


def scale_dtm(quantity):
    d_t_m = 3
    if quantity <= 3:
        return d_t_m
    elif quantity >= 7:
        return 0.
    return d_t_m - ((3 / 4) * (quantity - 3))


def empty_start_random(quantity):
    d_coef = 0.02
    d_c = 0.5
    d_t = 7
    d_t_m = scale_dtm(quantity)
    k = 1
    # rng = np.random.RandomState(3)
    random_uniform_scale = list()
    for x in range(2 * quantity):
        random_uniform_scale.append(uniform(loc=-d_c, scale=2 * d_c))
        random_uniform_scale.append(uniform(loc=0, scale=d_t))
        if x % 2 == 0:
            random_uniform_scale.append(uniform(loc=0, scale=k))
        else:
            random_uniform_scale.append(uniform(loc=0., scale=0.))
        for _ in range(8):
            random_uniform_scale.append(uniform(loc=-d_coef, scale=2 * d_coef))
    random_uniform_scale.append(uniform(loc=0, scale=d_t_m))

    values = [uni.rvs() for uni in random_uniform_scale]
    free_space = 7 - values[-1]
    sum_space = 0
    for x in range(2 * quantity):
        sum_space += values[x * 11 + 1]

    if sum_space > free_space:
        norm = free_space / sum_space
        for x in range(2 * quantity):
            values[x * 11 + 1] *= norm

    delete = 0
    for x in range(quantity):
        if values[2 * x * 11 + 1] < 0.1:
            delete += 0.1 - values[2 * x * 11 + 1]
            values[2 * x * 11 + 1] = 0.1

    for x in range(quantity):
        if values[(2 * x + 1) * 11 + 1] > delete / quantity:
            values[(2 * x + 1) * 11 + 1] -= delete / quantity
    values = np.array(values[:-1]).reshape(2 * quantity, 11)
    feature_done = np.zeros(2 * quantity)
    feature_done[-1] = 1

    return np.concatenate([values, feature_done.reshape(-1, 1)], axis=1)


def random_search_param(d, best_param, best_loss):
    # c1 = best_param['c1']
    # c2 = best_param['c2']
    # c3 = best_param['c3']
    # c4 = best_param['c4']
    param_grid = dict()
    rng = np.random.RandomState(0)
    # param_grid = {'c1': uniform(loc=c1 - d, scale=2 * d), 'c2': uniform(loc=c2 - d, scale=2 * d),
    #               'c3': uniform(loc=c3 - d, scale=2 * d), 'c4': uniform(loc=c4 - d, scale=2 * d)}
    [param_grid.update({f'{key}_{x}': uniform(loc=item[x] - d, scale=2 * d) for x in range(len(item))})
     for key, item in best_param.items()]

    param_list = list(ParameterSampler(param_grid, n_iter=10000,
                                       random_state=rng))
    for _param in tqdm(param_list):
        param = dict()
        for key, item in _param.items():
            param[key[:-2]] = param.get(key[:-2], list()) + [item]

        loss = main(**param)
        if loss <= best_loss:
            best_param = param
            best_loss = loss

    print('***********************************************')
    print(f'{best_loss=}, {best_param=}')
    print('***********************************************')
    return best_param, best_loss


def optimization_param_search(best_param):
    search_param = list()
    best_param = best_param.values()
    list(map(lambda x: search_param.extend(x), best_param))
    result = minimize(main, np.array(search_param))
    print(result)


def iterative_random_search(param, loss):
    for _d in tqdm([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
        param, loss = random_search_param(_d, param, loss)


if __name__ == '__main__':
    run = Run()
    main = run.main
    search = 'optimization'
    _l1 = [0.2747823174694503, 1., 1.54, 75.]
    _l2 = [0.13556582944950138, .5]
    _l3 = [-0.055209803982245384, 1., 1.67, 39.]
    _l4 = [-0.2568888474926888, 4.21639]
    _l1_coefs = [0.0, 0.009109298409282469, -0.03374649200850791,
                 0.01797256809388843, -0.0050513483804677005, 0.0,
                 0.0,
                 0.0]
    _l2_coefs = [0.0, -0.002874728268075267, -0.03373322938525211,
                 0.004205227876537139, -0.0001705765222318475,
                 0.0, 0.0, 0.0]
    _l3_coefs = [0.0, -0.0231369463217776, 0.011956554928461116,
                 -0.017782670650182023, 0.004077846642272649, 0.0,
                 0.0, 0.0]
    _l4_coefs = [0.] * 8
    _best_param = {'l1': _l1, 'l2': _l2, 'l3': _l3, 'l4': _l4, 'l1_coefs': _l1_coefs, 'l2_coefs': _l2_coefs,
                   'l3_coefs': _l3_coefs, 'l4_coefs': _l4_coefs}
    _best_loss = main(**_best_param)
    if search == 'random':
        iterative_random_search(_best_param, _best_loss)

    if search == 'optimization':
        optimization_param_search(_best_param)
