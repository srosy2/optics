isdark = False
from rayoptics.environment import *
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from rayoptics.util.misc_math import normalize
import re
import io
from contextlib import redirect_stdout


file = 'test_opt.roa'

# base_param
def my_calc_loss(path2model, params):
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

    def test_opt_model(load=False):
        if load:
            opm = open_model(f'{file}', info=True)
        else:
            opm = OpticalModel()
        sm = opm['seq_model']
        osp = opm['optical_spec']
        pm = opm['parax_model']
        em = opm['ele_model']
        pt = opm['part_tree']
        ar = opm['analysis_results']
        # osp.opt_model = true_osp.opt_model.copy()

        if not load:

            opm.system_spec.title = 'Cell Phone Lens - U.S. Patent 7,535,658'
            opm.system_spec.dimensions = 'mm'
            osp['pupil'] = PupilSpec(osp, key=['object', 'pupil'], value=2.5)
            osp['fov'] = FieldSpec(osp, key=['object', 'angle'], value=0.0, is_relative=False, flds=[0., 5., 10.,
                                                                                                     15., 20.],
                                   )
            osp['fov'].value = 0.0
            osp['wvls'] = WvlSpec([(470., 1.0), (650, 1.0)], ref_wl=1)
            opm.radius_mode = False

            sm.gaps[0].thi = 1e10

            sm.add_surface(params['lense_params_11'])
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=params['lense_params_11'][0])
                                                            #  , cc=0.0,
                                                            #  coefs=[0.0, 0.009109298409282469, -0.03374649200850791,
                                                            #         0.01797256809388843, -0.0050513483804677005, 0.0,
                                                            #         0.0,
                                                            #         0.0])
            # sm.ifcs[sm.cur_surface].interact_mode = 'transmit'
            # sm.set_stop()

            sm.add_surface(params['lense_params_12'])
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(params['lense_params_12'][0])
                                                            #  , cc=0.0,
                                                            #  coefs=[0.0, -0.002874728268075267, -0.03373322938525211,
                                                            #         0.004205227876537139, -0.0001705765222318475,
                                                            #         0.0, 0.0, 0.0])

            sm.add_surface(params['lense_params_21'])
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=params['lense_params_22'][0])
                                                            #  , cc=0.0,
                                                            #  coefs=[0.0, -0.0231369463217776, 0.011956554928461116,
                                                            #         -0.017782670650182023, 0.004077846642272649, 0.0,
                                                            #         0.0, 0.0])

            sm.add_surface(params['lense_params_22'])
            sm.ifcs[sm.cur_surface].profile = Spherical(c=params['lense_params_22'][0])
            sm.ifcs[-1].profile = EvenPolynomial(c=0.0, cc=0.0, coefs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            opm.update_model()
        sm.list_model()
        sm.list_surfaces()
        get_thichness(sm)
        if not load:
            opm.save_model(file)

        return opm

    # opm = open_model(f'{path2model}', info=True)
    opm = test_opt_model(load=False)
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
    psf = SpotDiagramFigure(opm)
    test_psf = psf.axis_data_array[field][0][0][0]
    test_psf[:, 1] = test_psf[:, 1] - np.mean(test_psf[:, 1])
    # plt.plot(test_psf[:, 0], test_psf[:, 1], 'o')
    # plt.rcParams['figure.figsize'] = (8, 8)
    # plt.show()

    fld, wvl, foc = osp.lookup_fld_wvl_focus(0)
    sm.list_model()
    sm.list_surfaces()
    efl = pm.opt_model['analysis_results']['parax_data'].fod.efl

    pm.first_order_data()
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
            test_psf[:, 1] = test_psf[:, 1] - np.mean(test_psf[:, 1])
            r_psf = np.sort(np.sqrt(test_psf[:, 0] ** 2 + test_psf[:, 1] ** 2))
            enclosed_energy = 100 * np.sum(r_psf <= radius_enclosed_energy_for_loss / 1e3) / len(test_psf[:, 0])
            loss_enclosed_energy = funct_loss_enclosed_energy(enclosed_energy, perc_max_enclosed_energy_for_loss,
                                                              perc_min_enclosed_energy_for_loss)
            loss_enclosed_energy_all = loss_enclosed_energy_all + loss_enclosed_energy

            dl = int(np.floor(len(test_psf[:, 0]) * perc_max_enclosed_energy_for_loss / 100))
            loss_rms = np.sqrt(np.sum((1e3 * r_psf[:dl]) ** 2) / dl)
            loss_rms_all = loss_rms_all + loss_rms

            # print(f'{idx_field=}, {idx_wavelength=}, {enclosed_energy=},  {loss_enclosed_energy=},  {loss_rms=}')
            temp = temp + 1
    loss_enclosed_energy_all = loss_enclosed_energy_all / temp
    loss_rms_all = loss_rms_all / temp
    loss = loss_focus + loss_FD + loss_total_length + loss_min_thickness + loss_min_thickness_air + loss_enclosed_energy_all + loss_rms_all
    # print(
    #     f'{loss_focus=}, {loss_FD=},  {loss_total_length=},  {loss_min_thickness=},  {loss_min_thickness_air=},  {loss_enclosed_energy_all=},  {loss_rms_all=}')
    # layout_plt0 = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,
    #                          do_draw_rays=True, do_paraxial_layout=False,
    #                          is_dark=isdark).plot()
    # print(f'final loss:{loss}')
    return (loss)
