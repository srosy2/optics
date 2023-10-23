import random

from rayoptics.environment import *
from optic_model import make_optic_surface_m1, make_optic_surface_m2, prepare_conf_from_arr
import numpy as np
import re
import io
from contextlib import redirect_stdout
import torch.nn as nn
import torch
from torch import Tensor
from features_loss import compute_third_order_cast
from search_param import empty_start_random
import ray
from ray.util.multiprocessing import Pool


class OpticEnv:
    """
    Есть 2 варианта начала, первый вариант мы загружаем подготовленный конфиг и
    считаем для него фичи, второй вариант мы загружаем quantity линз и воздуха между ними
    с начальными параметрами и считаем фичи для них
    Есть 2 варианта конфига которые подоются на step: 1. мы загружаем конфиг с точными параметрами
    для линз и воздуха и считаем фичи и лосс для них, 2. подается конфиг, где расписанна тольщина
    и что за объект линза или воздух и потом создается оптичиская линза с воздухом и линзами и с
    начальными параметрами и по ним считаются фичи и лосс
    """

    def __init__(self):
        self.num_envs = 1
        self.optic = None
        self.emb = nn.Embedding(2, 12)
        self.step_pass_m1 = 0
        self.step_pass_m2 = 0
        self.finish_m1 = 15
        self.finish_m2 = 5
        self.loss = None
        self.old_loss = dict()
        self.error_reward = -1
        ray.init(num_cpus=6, _temp_dir="/tmp",
                 include_dashboard=False, ignore_reinit_error=True)
        self.pool = Pool(6)

    def reset(self, model=1, quantity=2):
        self.step_pass_m1 = 0
        self.step_pass_m2 = 0
        self.optic = self.set_env_opm()
        if model == 1:
            conf = self.reset_conf_1()
            self.create_surf_from_conf_m1(conf)
            loss, feature_loss, reward = self.calc_loss_from_model(lins=2)
            feature_env = self.prepare_m1_feature_from_conf(conf)
            self.loss = loss
            return feature_env, feature_loss, loss

        elif model == 2:
            conf = self.reset_conf_2(quantity)
            self.create_surf_from_conf_m2(conf)
            loss, feature_loss, reward = self.calc_loss_from_model(lins=2)
            self.loss = loss
            feature_env = self.prepare_m2_feature_from_conf(conf)
            return feature_env, feature_loss, loss

        else:
            raise

    def reward(self, loss, lins, return_reward):
        reward = None
        old_loss = self.old_loss.get(f'lins_{lins}', None)
        self.old_loss[f'lins_{lins}'] = old_loss
        if return_reward:
            if old_loss is None:
                return 0
            elif loss > old_loss:
                return 1
            else:
                return -1
        return reward

    def set_env_opm(self):
        opm = OpticalModel()
        opm.system_spec.dimensions = 'mm'
        opm.radius_mode = False
        osp = opm['optical_spec']
        sm = opm['seq_model']
        osp['pupil'] = PupilSpec(osp, key=['object', 'pupil'], value=2.5)
        osp['fov'] = FieldSpec(osp, key=['object', 'angle'], value=0.0, is_relative=False, flds=[0., 5., 10.,
                                                                                                 15., 20.],
                               )
        osp['fov'].value = 0.0
        osp['wvls'] = WvlSpec([(470., 1.0), (650, 1.0)], ref_wl=1)

        sm.gaps[0].thi = 1e10
        sm.ifcs[-1].profile = EvenPolynomial(c=0.0, cc=0.0, coefs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        opm.update_model()
        return opm

    @staticmethod
    def reset_conf_1():
        conf = [{'type': 'linsa', 'profile': 'EvenPolynomial', 'surf': [0.2791158099269407, 1.0037479911346872,
                                                                        1.],
                 'coefs': [0.012868466936135371, 0.005063302914189549,
                           -0.03414342498052082, 0.027768776932613096, -0.010775548959399424, -0.008060620851500088,
                           0.009012318437796152, -0.0024075426547856284]},
                {'type': 'air', 'profile': 'EvenPolynomial', 'surf': [0.14207869840075518, 0.5096391331756961, 0.],
                 'coefs': [0.0112679970210799, -0.014238612981715795, -0.03185377276780764, 0.0015606715704450064,
                           0.005005532685212877, -0.003771280956461448, 0.0010002353156377192,
                           -9.240401608152898e-05]},
                {'type': 'linsa', 'profile': 'EvenPolynomial',
                 'surf': [-0.05626882503338093, 1.0030754980529646, 0.],
                 'coefs': [-0.002048763086163696, -0.0220739165445026, 0.004594937414782493, -0.01878687077411235,
                           0.009011927520039275, -0.0004573985435062217, -0.001679688857711538,
                           0.00048263606187476426]},
                {'type': 'air', 'profile': 'EvenPolynomial', 'surf': [-0.2563566127076245, 4.216404623070566, 0.],
                 'coefs': [0.] * 8}]
        return conf

    @staticmethod
    def reset_conf_2(quantity):
        conf = [{'type': 'linsa', 't': 7. / (2 * quantity)}, {'type': 'air', 't': quantity / (2 * quantity)}] * quantity
        return conf

    def prepare_m1_feature_from_conf(self, conf: list):
        all_features = list()
        for conf_surf in conf:
            features = list()
            features.extend(conf_surf['surf'])
            if conf_surf['profile'] == 'EvenPolynomial':
                features.extend(conf_surf['coefs'])
            elif conf_surf['profile'] == 'Spherical':
                features.extend([0.] * 8)
            all_features.append(features)
        feature_done = torch.zeros(len(all_features))
        feature_done[-1] = 1.
        # print(torch.cat(all_features).view(-1, 11))
        return torch.cat([torch.tensor(all_features), feature_done.view(-1, 1)], dim=1)

    def prepare_m2_feature_from_conf(self, conf: list):
        all_features = list()
        for conf_surf in conf:
            features = torch.tensor([0., conf_surf['t']] + [0.] * 10)
            if conf_surf['type'] == 'linsa':
                all_features.append(features + self.emb[0])
            elif conf_surf['type'] == 'air':
                all_features.append(features + self.emb[1])

        return torch.cat(all_features)

    def create_surf_from_conf_m1(self, conf: list):
        sm = self.optic['seq_model']
        for num, conf_surf in enumerate(conf):
            make_optic_surface_m1(sm, conf_surf)
            if num == 0:
                sm.set_stop()
        self.optic.update_model()

    def create_surf_from_conf_m2(self, conf):
        for conf_surf in conf:
            make_optic_surface_m2(self.optic['seq_model'], conf_surf)
        self.optic.update_model()

    def sample(self):
        probability = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        probability = probability / np.sum(probability)
        quantity = np.random.choice(list(range(1, 8)), 1, p=probability)
        return empty_start_random(quantity[0])

    def step_m1(self, actions, lins):
        self.step_pass_m1 += 1
        conf = prepare_conf_from_arr(actions)
        self.optic = self.set_env_opm()
        try:
            self.create_surf_from_conf_m1(conf)
        except:
            return None, None, None, self.error_reward
        try:
            loss, loss_feature, reward = self.calc_loss_from_model(return_reward=True, lins=lins)
        except:
            return None, None, None, self.error_reward
        feature_env, feature_loss = actions, loss_feature
        return feature_env, feature_loss, loss, reward

    def step_m2(self, conf):
        self.step_pass_m2 += 1
        self.optic = self.set_env_opm()
        self.create_surf_from_conf_m2(conf)
        loss, loss_feature = self.calc_loss_from_model()
        feature_env = torch.cat([self.prepare_m2_feature_from_conf(conf), loss_feature])
        return feature_env, loss, self.step_pass_m2 == self.finish_m2

    def calc_loss_from_model(self, lins, return_reward=False):
        efl_for_loss = 5  # mm
        fD_for_loss = 2.1
        total_length_for_loss = 7.0  # mm
        radius_enclosed_energy_for_loss = 50  # micron
        perc_max_enclosed_energy_for_loss = 80  # %
        perc_min_enclosed_energy_for_loss = 50  # %
        min_thickness_for_loss = 0.1  # mm
        min_thickness_air_for_loss = 0.0  # mm
        number_of_field = 5
        number_of_wavelength = 2

        def funct_loss_enclosed_energy(enclosed_energy, perc_max_enclosed_energy_for_loss,
                                       perc_min_enclosed_energy_for_loss):
            if enclosed_energy < perc_max_enclosed_energy_for_loss:
                if enclosed_energy < perc_min_enclosed_energy_for_loss:
                    loss_enclosed_energy = 1e3
                else:
                    loss_enclosed_energy = (perc_max_enclosed_energy_for_loss - enclosed_energy)
            else:
                loss_enclosed_energy = 0
            return loss_enclosed_energy

        def get_thichness(sm):
            f = io.StringIO()
            with redirect_stdout(f):
                sm.list_model()
            s = f.getvalue()
            rows = re.split(r"\n", s)
            thickness_list = []
            thickness_material_list = []
            thickness_air_list = []
            for row in rows[1:-1]:
                row = re.sub(r'\s+', r'!', row)
                values = re.split(r"!", row)
                if values[4] != 'air' and values[4] != '1':
                    thickness_material_list.append(float(values[3]))
                if values[4] == 'air' and values[4] != '1':
                    thickness_air_list.append(float(values[3]))
                thickness_list.append(float(values[3]))  # 3 - thickness, 2 - curvature, 4 - type of material
            number_of_surfaces = len(rows) - 2
            return thickness_list, thickness_material_list, thickness_air_list, number_of_surfaces

        # opm = open_model(f'{path2model}', info=True)
        opm = self.optic
        sm = opm['seq_model']
        osp = opm['optical_spec']
        pm = opm['parax_model']
        em = opm['ele_model']
        pt = opm['part_tree']
        ar = opm['analysis_results']

        # plt.figure(FigureClass=InteractiveLayout, opt_model=opm, is_dark=isdark).plot()

        efl = pm.opt_model['analysis_results']['parax_data'].fod.efl
        fD = pm.opt_model['analysis_results']['parax_data'].fod.fno

        ax_ray, pr_ray, fod = ar['parax_data']
        u_last = ax_ray[-1][mc.slp]
        central_wv = opm.nm_to_sys_units(sm.central_wavelength())
        n_last = pm.sys[-1][mc.indx]
        to_df = compute_third_order(opm)

        tr_df = to_df.apply(to.seidel_to_transverse_aberration, axis='columns', args=(n_last, u_last))
        distortion = tr_df.to_numpy()[-1, 5]

        field = 0
        psf = SpotDiagramFigure(opm, self.pool)
        test_psf = psf.axis_data_array[field][0][0][0]
        test_psf = np.array(test_psf)
        test_psf[:, 1] = test_psf[:, 1] - np.mean(test_psf[:, 1])

        fld, wvl, foc = osp.lookup_fld_wvl_focus(0)
        sm.list_model()
        sm.list_surfaces()
        efl = pm.opt_model['analysis_results']['parax_data'].fod.efl

        pm.first_order_data()
        features = compute_third_order_cast(opm)
        opm.update_model()

        # total_length=0
        # min_thickness=0.15
        if abs(efl - efl_for_loss) > 0.25:
            loss_focus = 1e2 * (efl - efl_for_loss) ** 2
        else:
            loss_focus = 0

        if abs(fD) >= fD_for_loss:
            loss_FD = 5 * 1e4 * (fD - fD_for_loss) ** 2
        else:
            loss_FD = 0

        thickness_list, thickness_material_list, thickness_air_list, number_of_surfaces = get_thichness(sm)
        # print(thickness_list, thickness_material_list, thickness_air_list, number_of_surfaces)
        total_length = np.sum(thickness_list[1:])
        min_thickness = np.min(thickness_material_list)
        min_thickness_air = np.min(thickness_air_list)
        if (total_length - total_length_for_loss) > 0:
            loss_total_length = 1e4 * (total_length - total_length_for_loss) ** 2
        else:
            loss_total_length = 0

        if min_thickness < min_thickness_for_loss:
            loss_min_thickness = 1e6 * (min_thickness - min_thickness_for_loss) ** 2
        else:
            loss_min_thickness = 0

        if min_thickness_air < min_thickness_air_for_loss:
            loss_min_thickness_air = 8e4 * (min_thickness_air - min_thickness_air_for_loss) ** 2
        else:
            loss_min_thickness_air = 0

        loss_enclosed_energy_all = 0
        loss_rms_all = 0
        temp = 0
        for idx_field in range(number_of_field):
            for idx_wavelength in range(number_of_wavelength):
                test_psf = psf.axis_data_array[idx_field][0][0][idx_wavelength]
                test_psf = np.array(test_psf)
                test_psf[:, 1] = test_psf[:, 1] - np.mean(test_psf[:, 1])
                r_psf = np.sort(np.sqrt(test_psf[:, 0] ** 2 + test_psf[:, 1] ** 2))
                enclosed_energy = 100 * np.sum(r_psf <= radius_enclosed_energy_for_loss / 1e3) / len(test_psf[:, 0])
                loss_enclosed_energy = funct_loss_enclosed_energy(enclosed_energy, perc_max_enclosed_energy_for_loss,
                                                                  perc_min_enclosed_energy_for_loss)
                loss_enclosed_energy_all = loss_enclosed_energy_all + loss_enclosed_energy

                dl = int(np.floor(len(test_psf[:, 0]) * perc_max_enclosed_energy_for_loss / 100))
                loss_rms = np.sqrt(np.sum((1e3 * r_psf[:dl]) ** 2) / dl)
                loss_rms_all = loss_rms_all + loss_rms
                temp = temp + 1
        loss_enclosed_energy_all = loss_enclosed_energy_all / temp
        loss_rms_all = loss_rms_all / temp
        loss = loss_focus + loss_FD + loss_total_length + loss_min_thickness + loss_min_thickness_air + \
               loss_enclosed_energy_all + loss_rms_all
        if loss != loss:
            raise Exception
        reward = self.reward(
            loss, lins, return_reward)
        return loss, features, reward
