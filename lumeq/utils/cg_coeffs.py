from lumeq import sys, np
from lumeq.utils import print_matrix

def print_cg_coeff(j1=0, m1=0, j2=0, m2=0, j3=0, m3=0, cg=0,
                   header=False, ic=0):
    if header:
        print('%9s %5s %5s %5s %5s %5s %8s' % ('j1', 'm1', 'j2', 'm2', 'j3', 'm3', 'CG'))

    if ic >= 0:
        #print('%2d: %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %9.4f'
        #        % (ic, j1, m1, j2, m2, j3, m3, cg))

        print('%2d:' % ic, end=' ')
        for a in [j1, m1, j2, m2, j3, m3]:
            if a.is_integer():
                print('%5d' % a, end=' ')
            else:
                a1, a2 = a.as_integer_ratio()
                print('%3d/%1d' % (a1, a2), end=' ')

        if isinstance(cg, list): # need to sqrt the first number
            a1, a2 = cg
            sign = '-' if np.sign(a2) == -1 else ''
            cg = a1 * a2**2
            if cg.is_integer():
                print('%8d' % cg)
            else:
                from fractions import Fraction
                cg = Fraction(cg).limit_denominator()
                a1, a2 = cg.numerator, cg.denominator
                print('%12s' % (sign+'sqrt('+str(a1)+'/'+str(a2)+')'))
        else:
            print('%9.4f' % cg)


def ladder_coeff(j, m, operator):
    r"""Return the ladder-operator coefficient for angular momentum ``j, m``.

    Notes:
        ``j`` can take values such as ``0, ±1/2, ±1, ±3/2, ±2, ...`` and
        ``m`` must lie in ``{-j, -j+1, ..., j-1, j}``.

        The ladder coefficients are
        ``C_± = sqrt(j (j + 1) - m (m ± 1))``
        and equivalently
        ``C_± = sqrt((j ∓ m) (j ± m + 1))``.

        They satisfy
        ``j_± |j m> = C_± |j (m ± 1)>``.
    """
    if operator == '+': operator = np.add
    elif operator == '-': operator = np.subtract
    else: raise ValueError('operator undefined')

    if m < -j or m > j:
        #raise ValueError('m should in [-j, j]')
        return 0.

    return np.sqrt(j*(j+1) - m*(operator(m, 1)))


def clebsch_gordan_coeff_direct(j1, m1, j2, m2, j3, m3, sqrt=False):
    r"""
    `j3 is \in {|j_1 - j_2|, \dots, j_1 + j_2}`
    `m3 = m_1 + m_2 \in {-J, -J+1, \dots, J-1, J}`
    """
    if (m1<-j1 or m1>j1) or (m2<-j2 or m2>j2) or (m3<-j3 or m3>j3) or (m1+m2 != m3):
        return 0.

    from scipy.special import factorial

    cg = (2.*j3+1)*factorial(j3+j1-j2) * factorial(j3-j1+j2) * factorial(j1+j2-j3) / factorial(j1+j2+j3+1)
    cg *= factorial(j3+m3) * factorial(j3-m3) * factorial(j1+m1) * factorial(j1-m1) * factorial(j2+m2) * factorial(j2-m2)
    #cg = np.sqrt(cg)

    a = np.array([j1 + j2 - j3, j1 - m1, j2 + m2, j3 - j2 + m1, j3 - j1 - m2])

    kmin, kmax = max(0., -np.min(a[3:])), np.min(a[:3])
    if kmax < kmin: return 0.

    c = 0.
    for k in np.arange(kmin, kmax+1):
        c1 = a - k
        c1[3:] = a[3:] + k

        c1 = factorial(k) * np.prod(factorial(c1))
        c += (-1)**k / c1

    if sqrt:
        return np.sqrt(cg) * c
    else:
        return [cg, c]


def clebsch_gordan_coeff_recur(j1, m1, j2, m2):
    r"""
    j3 is in `{|j_1 - j_2|, \dots, j_1 + j_2}`
    `m3 = m_1 + m_2 \in {-J, -J+1, \dots, J-1, J}`
    """
    m3 = m1 + m2
    for j3 in range(abs(j1-j2), j1+j2+1):
        return


if __name__ == '__main__':
    #for j in range(3):
    #    for m in range(-j, j+1):
    #        cp = ladder_coeff(j, m, '+')
    #        cm = ladder_coeff(j, m, '-')
    #        print('j: %2d m: %2d cp: %8.4f cm: %8.4f' % (j, m, cp, cm))


    print_cg_coeff(header=True, ic=-1)

    ic = 0
    j1 = 1.
    j2 = 1.
    for m1 in np.arange(-j1, j1+1):
        for m2 in np.arange(-j2, j2+1):
            m3 = m1 + m2
            for j3 in np.arange(abs(j1-j2), j1+j2+1):
                cg = clebsch_gordan_coeff_direct(j1, m1, j2, m2, j3, m3)
                if isinstance(cg, list) or abs(cg) > 1e-8:
                    print_cg_coeff(j1, m1, j2, m2, j3, m3, cg, False, ic)
                    ic += 1
