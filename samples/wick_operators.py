from lumeq import sys
from lumeq.utils.wick_contraction import sqo_evaluation, commutator
from lumeq.utils.wick_contraction import contract_hamil_delta, print_math
from lumeq import itertools

if __name__ == '__main__':
    latex = True
    colors = ['red', 'blue', 'wickgreen', 'black', 'wickorange', 'purple']

    # 1e and 2e Hamiltonian terms
    h1 = 'p^dagger q'
    h2 = 'p^dagger r^dagger ell q'
    h2_ops = h2.split()
    exceptions0 = [(h2_ops[0], h2_ops[2]), (h2_ops[0], h2_ops[3]), (h2_ops[1], h2_ops[2]), (h2_ops[1], h2_ops[3])]

    # open-shell excitation operators
    Tia = 'i_alpha^dagger a_alpha' # bra side
    Tbj = 'b_alpha^dagger j_alpha' # ket side
    exceptions = exceptions0 + [tuple(Tia.split()), tuple(Tbj.split())]

    for h in [h1, h2]:
        term_type = '1e' if h == h1 else '2e'
        title = f'Open-shell excited-state {term_type} term contractions:'
        sqo_evaluation(Tia, h, Tbj, exceptions=exceptions, title=title,
                       colors=colors, latex=latex, symmetry=True, exchange=True)

    # spin-flip excitation operators diagonal terms
    Tst_mr = ['s_beta^dagger t_alpha', 's_alpha^dagger t_beta'] # Ms=+1,-1 reference bra
    Tts_mr = ['t_alpha^dagger s_beta', 't_beta^dagger s_alpha'] # Ms=+1,-1 reference ket
    Tia_mr = ['i_beta^dagger a_alpha', 'i_alpha^dagger a_beta'] # bra side sf excitation
    Tbj_mr = ['b_alpha^dagger j_beta', 'b_beta^dagger j_alpha'] # ket side sf excitation
    #Tst_mr = ['s_bar-sigma^dagger t_sigma', 's_sigma^dagger t_bar-sigma'] # Ms=+1,-1 reference bra
    #Tts_mr = ['t_sigma^dagger s_bar-sigma', 't_bar-sigma^dagger s_sigma'] # Ms=+1,-1 reference ket
    #Tia_mr = ['i_bar-sigma^dagger a_sigma', 'i_sigma^dagger a_bar-sigma'] # bra side sf excitation
    #Tbj_mr = ['b_sigma^dagger j_bar-sigma', 'b_bar-sigma^dagger j_sigma'] # ket side sf excitation

    exceptions = exceptions0
    exceptions += [tuple(Tst.split()) for Tst in Tst_mr]
    exceptions += [tuple(Tts.split()) for Tts in Tts_mr]
    exceptions += [tuple(Tia.split()) for Tia in Tia_mr]
    exceptions += [tuple(Tbj.split()) for Tbj in Tbj_mr]

    #for o1 in ['s', 't', 'i', 'b']:
    #    for o2 in ['t', 's', 'a', 'j']:
    #        exceptions += [(o1+'_bar-sigma^dagger', o2+'_sigma'), (o1+'_sigma^dagger', o2+'_bar-sigma')]


    f = 0
    fb = (f+1) % 2
    for iii, h in enumerate([h1, h2]):
        term_type = 'metric' if h == '' else ('1e' if h == h1 else '2e')
        title = f'Spin-flip excited-state {term_type} term contractions:'

        Tst = Tst_mr[f]
        Tts = Tts_mr[f]
        Tia = Tia_mr[f]
        Tbj = Tbj_mr[f]

        middles, factors = commutator(Tia, h, Tbj)
        #sqo_evaluation(Tst, middle, Tts, exceptions=exceptions, title=title,
        #               hamiltonian=h, latex=latex, diagram=False)

        Tst = Tst_mr[f]
        Tts = Tts_mr[fb]
        middles = [[Tia_mr[fb], h, Tbj_mr[f]], [Tbj_mr[fb], h, Tia_mr[f]],
                   [Tia_mr[fb], Tbj_mr[fb], h], [Tbj_mr[fb], Tia_mr[fb], h],
                   [h, Tia_mr[f], Tbj_mr[f]], [h, Tbj_mr[f], Tia_mr[f]]]

        collect_data = []
        for middle, factor in zip(middles, factors):
            middle = ' '.join(middle)
            print(f'Commutator term: {factor} * {Tst} {middle} {Tts}')
            data = sqo_evaluation(Tst, middle, Tts, exceptions=exceptions,
                                  title=title,
                                  hamiltonian=h, latex=latex, diagram=False,
                                  delta_style='^_', symmetry=True, exchange=True)

            collect_data.append(data[1])

        if iii == 1:
            # combine same terms
            print('Combining same terms...')
            strings = contract_hamil_delta(h, [*collect_data[2], *collect_data[3]], symmetry=True, exchange=True)
            print_math(' '.join(strings), 'anticommutator terms', latex=latex)
            strings = contract_hamil_delta(h, [*collect_data[4], *collect_data[5]], symmetry=True, exchange=True)
            print_math(' '.join(strings), 'anticommutator terms', latex=latex)
