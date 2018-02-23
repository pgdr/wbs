#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy.optimize as opt

class simulator(object):
    def __init__(self, employees, projects, fte, wbs):
        self.employees = employees
        self.projects = projects
        self.fte = fte
        self.wbs = wbs

        self.N = len(employees)
        self.M = len(projects)

        self.bounds = self._gen_bounds()

        total_fte = sum([fte[p] for p in projects])
        if abs(total_fte - len(employees)) >= .5:
            print('warning: %.2f ftes for %d employees' % (total_fte, self.N))

    @property
    def dims(self):
        return [.0] * self.N * self.M

    def func(self, point):
        mx = np.reshape(np.array(point), (self.N, self.M))
        emp_pen = 0.  # penalty per employee
        for i, e in enumerate(self.employees):
            emp = (1. - sum(mx[i]))**2
            emp_pen += emp

        # overtime penalty
        for i, e in enumerate(self.employees):
            emp = (1. - sum(mx[i]))**2
            if emp > 1.:
                emp_pen += emp - 1.

        pro_pen = 0.  # penalty per project
        for j, p in enumerate(self.projects):
            fte = self.fte[p]
            pro = (sum([mx[i][j] for i in range(self.N)]) - fte)**2
            pro_pen += pro

        return emp_pen + pro_pen

    def _gen_bounds(self):
        bounds = [(0., 2.) for _ in range(self.N * self.M)]
        for e_i, e in enumerate(self.employees):
            i = e_i * len(self.projects)
            for j, p in enumerate(self.projects):
                if self.wbs[e] is not None and p in self.wbs[e]:
                    bounds[i + j] = self.wbs[e][p]
                else:
                    # with this, you can only work on specified projects
                    # commented out so everyone can work on anything
                    # bounds[i + j] = (0.,0.)
                    pass
        return bounds


def print_result(res, employees, projects):
    floatstr = lambda x: '%.2f' % x
    lstr = lambda x: str(x).ljust(10)

    mx = np.reshape(np.array(res.x), (len(employees), len(projects)))

    print(' ' * 20, ' '.join(map(lstr, projects)))
    for i, e in enumerate(employees):
        e_str = '%s (%.2f)' % (e.ljust(10), sum(mx[i]))
        print('%s %s' % (e_str.ljust(20),
                         ' '.join(map(lstr, map(floatstr, mx[i])))))

    print('Penalty: %.2f' % res.fun)


def main(settings):
    employees = sorted(settings['employees'].keys())
    projects = sorted(settings['projects'].keys())
    sim = simulator(employees, projects, settings['projects'],
                    settings['employees'])

    res = opt.minimize(sim.func, x0=sim.dims, bounds=sim.bounds)
    print_result(res, employees, projects)


if __name__ == '__main__':
    from sys import argv
    if len(argv) != 2:
        exit('usage: wbs.py sib.yml')
    import yaml
    with open(argv[1], 'r') as fyml:
        settings = yaml.load(fyml)
    main(settings)
