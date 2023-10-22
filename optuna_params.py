import optuna
from my_ray_optics_criteria_ITMO import my_calc_loss
import warnings 
from threading import Thread

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calc_n(k):
    return 1.54 * k + 1.67 * (1-k)

def calc_abbe(k):
    return 75 * k + 39 * (1-k)

def objective(trial):
    
    radius_of_curvature_11 = trial.suggest_float('radius_of_curvature_11', -10, 10)
    surface_thickness_11 = trial.suggest_float('surface_thickness_11', 0.1, 1)
    substance_concentration_1 = trial.suggest_float('substance_concentration_1', 0.0, 1)
    
    lense_params_11 = [
        radius_of_curvature_11, 
        surface_thickness_11,
        calc_n(substance_concentration_1),
        calc_abbe(substance_concentration_1),
    ]
    
    radius_of_curvature_12 = trial.suggest_float('radius_of_curvature_12', -10, 10)
    surface_thickness_12 = trial.suggest_float('surface_thickness_12', 0.1, 1)
    
    lense_params_12 = [
        radius_of_curvature_12, 
        surface_thickness_12,
    ]
    
    radius_of_curvature_21 = trial.suggest_float('radius_of_curvature_21', -10, 10)
    surface_thickness_21 = trial.suggest_float('surface_thickness_21', 0.1, 1)
    substance_concentration_2 = trial.suggest_float('substance_concentration_2', 0.0, 1)
    
    lense_params_21 = [
        radius_of_curvature_21, 
        surface_thickness_21,
        calc_n(substance_concentration_2),
        calc_abbe(substance_concentration_2),
    ]
    
    radius_of_curvature_22 = trial.suggest_float('radius_of_curvature_22', -10, 10)
    surface_thickness_22 = trial.suggest_float('surface_thickness_22', 0.1, 1)
    
    lense_params_22 = [
        radius_of_curvature_22, 
        surface_thickness_22,
    ]

    params = {}
    params['lense_params_11'] = lense_params_11
    params['lense_params_12'] = lense_params_12
    params['lense_params_21'] = lense_params_21
    params['lense_params_22'] = lense_params_22

    thickness_sum = 0
    for i in params.values():
        thickness_sum += i[1]
 
    if thickness_sum > 7:
      return float("inf")
    
    path2model='test.roa'
    
    try:
        loss = my_calc_loss(path2model, params)
    except:
        return float("inf")
    return loss


if __name__ == '__main__':
    study = optuna.create_study(
        study_name='optics', 
        direction='minimize',
        storage='postgresql://postgres:123@localhost:5432/optics',
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=20000)

    trial = study.best_trial
    print('Value: ', trial.value)
    print('Params: ', trial.params)