from rayoptics.environment import *


# def test_opt_model(load=False, file='test_opt.roa', c1=0.2747823174694503, c2=0.13556582944950138,
#                    c3=-0.055209803982245384, c4=-0.2568888474926888):
def test_opt_model(load=False, file='test_opt.roa', l1=None, l2=None,
                   l3=None, l4=None, l1_coefs=None, l2_coefs=None, l3_coefs=None, l4_coefs=None):
    if l4_coefs is None:
        l4_coefs = [0.] * 8
    if l3_coefs is None:
        l3_coefs = [0.0, -0.0231369463217776, 0.011956554928461116,
                    -0.017782670650182023, 0.004077846642272649, 0.0,
                    0.0, 0.0]
    if l2_coefs is None:
        l2_coefs = [0.0, -0.002874728268075267, -0.03373322938525211,
                    0.004205227876537139, -0.0001705765222318475,
                    0.0, 0.0, 0.0]
    if l1_coefs is None:
        l1_coefs = [0.0, 0.009109298409282469, -0.03374649200850791,
                    0.01797256809388843, -0.0050513483804677005, 0.0,
                    0.0,
                    0.0]
    if l4 is None:
        l4 = [-0.2568888474926888, 4.21639]
    if l3 is None:
        l3 = [-0.055209803982245384, 1., 1.67, 39.]
    if l2 is None:
        l2 = [0.13556582944950138, .5]
    if l1 is None:
        l1 = [0.2747823174694503, 1., 1.54, 75.]
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

        sm.add_surface(l1)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=l1[0], cc=0.0,
                                                         coefs=l1_coefs)
        sm.ifcs[sm.cur_surface].interact_mode = 'transmit'
        sm.set_stop()

        sm.add_surface(l2)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=l2[0], cc=0.0,
                                                         coefs=l2_coefs)

        sm.add_surface(l3)
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=l3[0], cc=0.0,
                                                         coefs=l3_coefs)

        sm.add_surface(l4)
        # sm.ifcs[sm.cur_surface].profile = Spherical(c=l4[0])
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=l4[0], cc=0.,
                                                         coefs=l4_coefs)
        sm.ifcs[-1].profile = EvenPolynomial(c=0.0, cc=0.0, coefs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        opm.update_model()
    sm.list_model()
    sm.list_surfaces()
    if not load:
        opm.save_model(file)

    return opm


def make_optic_surface_m1(sm, conf: dict):
    sm.add_surface(conf['surf'])
    if conf['profile'] == 'EvenPolynomial':
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=conf['surf'][0], cc=0.0,
                                                         coefs=conf['coefs'])
    elif conf['profile'] == 'Spherical':
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=conf['surf'][0])
    return sm


def make_optic_surface_m2(sm, conf: dict):
    if conf['type'] == 'linsa':
        sm.add_surface([0., conf['t'], 0., 0.])
    elif conf['type'] == 'air':
        sm.add_surface([0., conf['t']])

    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=0., cc=0., coefs=[0.] * 8)
    return sm
