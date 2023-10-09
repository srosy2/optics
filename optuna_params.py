import optuna
from my_ray_optics_criteria_ITMO import my_calc_loss
import warnings 
from threading import Thread

warnings.filterwarnings("ignore", category=RuntimeWarning)

def objective(trial):
    efl_for_loss = trial.suggest_float('efl_for_loss', 0.1, 10.0)
    fD_for_loss = trial.suggest_float('fD_for_loss', 0.1, 5.0)
    total_length_for_loss = trial.suggest_float('total_length_for_loss', 0.1, 10)
    radius_enclosed_energy_for_loss = trial.suggest_float('radius_enclosed_energy_for_loss', 1, 100)
    perc_max_enclosed_energy_for_loss = trial.suggest_float('perc_max_enclosed_energy_for_loss', 10, 100)
    perc_min_enclosed_energy_for_loss = trial.suggest_float('perc_min_enclosed_energy_for_loss', 10, 100)
    min_thickness_for_loss = trial.suggest_float('min_thickness_for_loss', 0.1, 10)
    min_thickness_air_for_loss = trial.suggest_float('min_thickness_air_for_loss', 0.1, 10)
    number_of_field = 5
    number_of_wavelength = 2

    params = {}
    params['efl_for_loss'] = efl_for_loss
    params['fD_for_loss'] = fD_for_loss
    params['total_length_for_loss'] = total_length_for_loss
    params['radius_enclosed_energy_for_loss'] = radius_enclosed_energy_for_loss
    params['perc_max_enclosed_energy_for_loss'] = perc_max_enclosed_energy_for_loss
    params['perc_min_enclosed_energy_for_loss'] = perc_min_enclosed_energy_for_loss
    params['min_thickness_for_loss'] = min_thickness_for_loss
    params['min_thickness_air_for_loss'] = min_thickness_air_for_loss
    params['number_of_field'] = number_of_field
    params['number_of_wavelength'] = number_of_wavelength

    path2model='test.roa'
    loss = my_calc_loss(path2model, params)

    return loss



if __name__ == '__main__':
    study = optuna.create_study(
        study_name='optics', 
        direction='minimize',
        storage='postgresql://postgres:123@localhost:5432/optics',
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=10)

    trial = study.best_trial
    print('Value: ', trial.value)
    print('Params: ', trial.params)