from ray_optics_criteria_ITMO import calc_loss
import time


class Run:
    def __init__(self):
        self.best_loss = float('inf')

    def main(self, *args, **kwargs):
        t0 = time.time()
        path2model = 'test.roa'  # insert the path to ray-optics model, for example 'C:/test.roa'
        print(args, kwargs)
        if args:
            args = list(args[0])
            arg_len = len(args) // 11
            kwargs = dict()
            start = 0
            for x in range(1, arg_len + 1):
                kwargs.update({f'l{x}': args[start: start + 2 * (2**(x % 2))]})
                start += 2 * (2**(x % 2))
            # l_value = [0, 4, 6, 10, 12, 16, 18]
            # kwargs = {f'l{1 + x}': args[l_value[x]: l_value[x + 1]] for x in range(len(l_value) - 1)}
            args = args[start:]
            for x in range(arg_len):
                kwargs.update({f'l{1 + x}_coefs': args[x * 8: (x + 1) * 8]})
            args = ()
            print(kwargs)
        loss = calc_loss(path2model, *args, **kwargs)
        if loss < self.best_loss:
            self.best_loss = loss
            with open('best_4.txt', 'a') as file:
                file.write(f'{loss=}, best_param={kwargs} \n')
        elapsed_time = time.time() - t0
        print(f'{loss=}, {elapsed_time=} sec')
        return loss


if __name__ == '__main__':
    run = Run()
    best_param = {'l1': [0.2791158099269407, 1.0037479911346872, 1.5261812896231235, 75.00008653474656],
                  'l2': [0.14207869840075518, 0.5096391331756961],
                  'l3': [-0.05626882503338093, 1.0030754980529646, 1.6777742569666922, 39.00015787264238],
                  'l4': [-0.2563566127076245, 4.216404623070566],
                  'l1_coefs': [0.012868466936135371, 0.005063302914189549, -0.03414342498052082, 0.027768776932613096,
                               -0.010775548959399424, -0.008060620851500088, 0.009012318437796152,
                               -0.0024075426547856284],
                  'l2_coefs': [0.0112679970210799, -0.014238612981715795, -0.03185377276780764, 0.0015606715704450064,
                               0.005005532685212877, -0.003771280956461448, 0.0010002353156377192,
                               -9.240401608152898e-05],
                  'l3_coefs': [-0.002048763086163696, -0.0220739165445026, 0.004594937414782493, -0.01878687077411235,
                               0.009011927520039275, -0.0004573985435062217, -0.001679688857711538,
                               0.00048263606187476426]}
    run.main(**best_param)
