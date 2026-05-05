from lumeq import sys, np, itertools
#from lumeq.spins import sympy
from lumeq.plot import get_plot_colors

import unicodedata
from collections import defaultdict

"""
Generate Wick contraction pairs and patterns from Fermionic second-quantization operator strings.
The operator here has the form like 'a^', 'i', 'b†', 'j_α', 'c_b^', 'k_beta',
representing creation and annihilation operators (with spin labels).
Remember to leave the dagger symbol at the end of the operator string.
"""

def is_creator(operator):
    r"""Check if the operator is a creation operator by dagger symbol."""
    return operator.endswith(('^', '†', '^dagger'))


def is_annihilator(operator):
    r"""Check if the operator is an annihilation operator."""
    return not is_creator(operator)


def remove_dagger(operator):
    r"""Remove the dagger symbol from the operator string."""
    return operator.split('^')[0].split('†')[0]


def get_orbital_index(operator):
    r"""Extract the orbital index from the operator string."""
    return remove_dagger(operator).split('_')[0]


_known_spins = ['alpha', 'beta', None] # known spin labels
def get_spin_label(operator):
    r"""Extract the spin label from the operator string."""
    if '_' in operator:
        spin = remove_dagger(operator.split('_')[1])
        if spin in ('a', 'alpha', 'up', 'α'):
            return 'alpha'
        elif spin in ('b', 'beta', 'down', 'β'):
            return 'beta'
        else:
            return spin
    return ''


def get_spin_orbital_index(operator):
    r"""Get the spin orbital index from the operator string."""
    return get_orbital_index(operator) + '_' + str(get_spin_label(operator))


def is_same_pattern(contraction1, contraction2):
    r"""Check if two contraction patterns are the same, ignoring order."""
    set1 = {frozenset(pair) for pair in contraction1}
    set2 = {frozenset(pair) for pair in contraction2}
    return set1 == set2


def has_pattern(contraction_list, target_pattern):
    r"""Check if a target contraction pattern exists in a list of contraction patterns.

    Args:
        contraction_list (list): List of contraction patterns, where each pattern is a
            list of pairs.
        target_pattern (list): Contraction pattern to search for.

    Returns:
        exists (bool): True if ``target_pattern`` exists in ``contraction_list``.
    """
    for pattern in contraction_list:
        if is_same_pattern(pattern, target_pattern):
            return True
    return False


def get_list(operators):
    r"""
    Convert input operators string to a list.

    Args:
        operators (list or str): Operator list, or a string separated by spaces or commas.

    Returns:
        operators_list (list): List of operator strings.
    """
    if isinstance(operators, str):
        operators = operators.replace(',', ' ').split()
    return operators


def wick_pairs(operators, exceptions=[], index=False):
    r"""
    Pick all the possible Wick contraction pairs from a string of creation and annihilation operators.

    Args:
        operators (list or str): A list of strings representing creation and
            annihilation operators with optional spin labels, or a string
            separated by spaces or commas.
            Eg. ['a^', 'i', 'b†', 'j_α', 'c_b^', 'k_beta']
            or "a^ i b† j_α c_b^ k_beta"
        exceptions (tuple or list, optional): Operator pairs that should not be contracted.
        index (bool, optional): If True, return operator indices instead of operator strings.

    Returns:
        pairs (list): All possible contraction pairs of the operators.
    """
    operators = get_list(operators)

    if isinstance(exceptions, tuple):
        exceptions = [exceptions]

    n = len(operators)
    if n % 2 != 0:
        raise ValueError("The number of operators must be even for Wick contraction.")

    creators = np.array([is_creator(op) for op in operators], dtype=bool)
    annihilators = ~creators
    spins = [get_spin_label(op) for op in operators]

    pairs = []

    for i in range(n):
        op_i = operators[i]
        cr_i = creators[i]
        an_i = annihilators[i]
        sp_i = spins[i]

        for j in range(i + 1, n):
            op_j = operators[j]
            cr_j = creators[j]
            an_j = annihilators[j]
            sp_j = spins[j]

            # exclude exception pairs of operators
            skip = [(op_i, op_j) == exc or (op_j, op_i) == exc for exc in exceptions]
            if np.any(skip):
                continue

            if (is_creator(op_i) and is_annihilator(op_j)) or (is_annihilator(op_i) and is_creator(op_j)):
                if sp_i in _known_spins and sp_j in _known_spins:
                    if sp_i == sp_j: # both known spins and should match
                        pairs.append((i, j))
                else:
                    pairs.append((i, j))

    if not index: # convert indices back to operator strings
        pairs = index_to_operators(operators, pairs)

    return pairs


def index_to_operators(operators, pairs_index):
    r"""
    Convert pairs of operator indices to operator strings.

    Args:
        operators (list): Operator strings.
        pairs_index (list): Tuple pairs of operator indices.

    Returns:
        pairs (list): Tuple pairs of operator strings.
    """
    pairs = [(operators[i], operators[j]) for (i, j) in pairs_index]
    return pairs


def wick_contraction(operators, pairs, expand=True):
    r"""
    Perform Wick contraction on the given pairs of operators.

    Args:
        operators (list): Operator strings.
        pairs (list): Tuple pairs of indices to be contracted.
        expand (bool, optional): If True, return product patterns. Otherwise return a
            dictionary of lists of contraction patterns.

    Returns:
        contractions (list): Contraction patterns.
    """
    operators = get_list(operators)
    n_op = len(operators) # total number of operators
    # pairs of operator strings or indices
    is_index = True if isinstance(pairs[0][0], int) else False

    contractions = defaultdict(list) # set up empty lists for keys not in dict
    for (op1, op2) in pairs:
        contractions[op1].append(op2)

    if expand:
        key_list = list(contractions.keys())
        value_list = [contractions[k] for k in key_list]
        n_keys = len(key_list)
        dim = [len(v) for v in value_list]

        contractions = []
        for idx in itertools.product(*[range(d) for d in dim]):
            #c_list = [(key_list[i], value_list[i][idx[i]]) for i in range(n_keys)]
            c_list, c_flat = [], []
            for i in range(n_keys):
                a, b = key_list[i], value_list[i][idx[i]]
                if a not in c_flat and b not in c_flat:
                    c_flat += [a, b]
                    if is_index:
                        c_list.append((a, b, operators[a], operators[b]))
                    else:
                        c_list.append((a, b)) # operator strings

            if len(c_flat) == n_op and not has_pattern(contractions, c_list):
                #print(c_list)
                contractions.append(c_list)

    return contractions[::-1] # reverse the order for convenience


def delta_format(style):
    r"""Get the write and parse functions for delta function formatting based on the specified style."""
    if style == 'upper_lower' or ('^' in style and '_' in style):
        def write(orb1, orb2, reverse=False):
            if reverse:
                return f'delta^{{{orb2}}}_{{{orb1}}}' + f'(1-n_{{{orb2}}})'
            else:
                return f'delta^{{{orb1}}}_{{{orb2}}}' + f'n_{{{orb2}}}'

        def parse(delta_str):
            content = delta_str.split('{')
            orb1, orb2 = content[1].split('}')[0], content[2].split('}')[0]
            o = delta_str.find('}', delta_str.index('}') + 1)
            occupancy = delta_str[o+1:]
            return orb1, orb2, occupancy

    elif style in {'lower', 'comma', '_'}:
        def write(orb1, orb2, reverse=False):
            if reverse:
                return f'delta_{{{orb2},{orb1}}}' + f'(1-n_{{{orb2}}})'
            else:
                return f'delta_{{{orb1},{orb2}}}' + f'n_{{{orb2}}}'

        def parse(delta_str):
            content = delta_str[delta_str.index('{')+1:delta_str.index('}')]
            orb1, orb2 = content.split(',')
            return orb1, orb2, delta_str[delta_str.index('}')+1:]

    return write, parse


def wick_delta(contractions, delta_style='comma'):
    r"""
    Convert contraction pairs into delta functions.

    Args:
        contractions (list): Contraction patterns.
        delta_style (str, optional): Style for formatting delta functions.
            Options include 'upper_lower', 'lower', 'comma', etc.

    Returns:
        deltas (list): Delta-function strings representing the contractions.
    """
    if isinstance(contractions[0], list): # loop over multiple contraction patterns
        return [wick_delta(pair, delta_style) for pair in contractions]

    if len(contractions[0]) == 2:
        raise ValueError(f'Contractions should have indices as well to determine signs.\n' +
                         f'Use wick_pairs() with index=True option before wick_contraction.')


    delta_write = delta_format(delta_style)[0]

    index = []
    deltas = []
    for (i1, i2, op1, op2) in contractions:
        orb1 = get_spin_orbital_index(op1)
        orb2 = get_spin_orbital_index(op2)
        index.append((i1, i2))
        if orb1 != orb2: # only add delta if orbitals are different
            deltas.append(delta_write(orb1, orb2, is_creator(op2)))

    sign = find_delta_sign(index, dtype=str)
    return sign + ' '.join(deltas)


def find_delta_sign(contractions_index, dtype=str):
    r"""
    Determine the sign of the contraction based on the number of crossings.

    Args:
        contractions_index (list): List of contraction pairs.
        dtype (type): Return type, typically ``str`` or ``int``.

    Returns:
        sign (str or int): ``'+'`` or ``'-'`` (or ``+1`` / ``-1``) depending on the
            number of crossings.
    """
    sign = 1
    n = len(contractions_index)
    for i, (i1, i2) in enumerate(contractions_index):
        if i1 > i2: # ensure i1 < i2
            i1, i2 = i2, i1
        for _, (j1, j2) in enumerate(contractions_index[i:]):
            if j1 > j2:
                j1, j2 = j2, j1

            if i1 < j1 < i2 < j2: # crossing detected
                sign *= -1

    if dtype == str:
        return '+' if sign == 1 else '-'
    else:
        return sign


def contract_hamil_delta(hamiltonian, deltas, symmetry=False, exchange=False):
    r"""
    Contract the Hamiltonian operator with delta functions.

    Args:
        hamiltonian (str): Hamiltonian operator string.
        deltas (list): Delta-function strings.
        symmetry (bool, optional): If True, apply symmetry to the two-electron integrals
        exchange (bool, optional): If True, combine exchange-integral pairs as
            antisymmetrized two-electron integrals.

    Returns:
        strings (list): Contracted Hamiltonian terms as strings.
    """
    if isinstance(deltas, str): # single set of contraction pattern
        deltas = [deltas]

    delta_style = 'comma' if ',' in deltas[0] else 'upper_lower'
    delta_parse = delta_format(delta_style)[1]

    hamiltonian = get_list(hamiltonian)
    n_hs = len(hamiltonian)
    if n_hs == 0: # overlap return the deltas directly
        return [d.replace(',', ' ') for d in deltas]
    h = 'h' if n_hs == 2 else 'g'

    strings = []
    for delta in deltas:
        sign = delta[0]
        delta_terms = delta[1:].split()
        h_terms = hamiltonian.copy()

        # find matching indices in Hamiltonian terms
        for i, h_op in enumerate(h_terms):
            h_orb = get_spin_orbital_index(h_op)

            for j, d in enumerate(delta_terms):
                if not d.startswith('delta'):
                    continue

                # extract orbital indices from delta function
                orb1, orb2, occupancy = delta_parse(d)

                if h_orb == orb1:
                    h_terms[i] = orb2
                    delta_terms[j] = occupancy.replace(orb1, orb2)  # mark for removal
                elif h_orb == orb2:
                    h_terms[i] = orb1
                    delta_terms[j] = occupancy.replace(orb2, orb1)  # mark for removal

        # reorder hamiltonian
        #h_contracted = h_terms
        h_contracted = [''] * n_hs
        for i in range(n_hs//2):
            h_contracted[2*i], h_contracted[2*i+1] = h_terms[i], h_terms[-1-i]
            if i in {1,2}: h_contracted[2*i-1] += ';' # separate electrons
        h_contracted = h + '_{' + ' '.join(h_contracted) + '}'

        term_str = ' '.join(delta_terms) + ' ' + h_contracted
        # replace commas of delta with spaces, and semicolons with commas
        term_str = term_str.replace(',', ' ').replace(';', ',')
        strings.append(f'{sign} {term_str}\n')

    if symmetry or exchange:
        return combine_same_terms(strings, exchange=exchange)
    return strings


def combine_same_terms(contracted_strings, exchange=False):
    r"""
    Apply symmetry to two-electron integrals in the contracted strings.

    Args:
        contracted_strings (list): Operator strings.
        exchange (bool, optional): If True, combine opposite-signed terms where
            swapping the second and fourth indices of one integral gives the other
            integral, and write the result as ``\tilde{g}``.

    Returns:
        sym_strings (list): Operator strings with symmetry applied.
    """
    def get_g_key(left_key, right_key):
        return tuple(sorted((left_key, right_key)))

    def get_exchange_key(key):
        if len(key) != 2 or len(key[0]) != 2 or len(key[1]) != 2:
            return None
        left_key, right_key = key
        exchanged_left = (left_key[0], right_key[1])
        exchanged_right = (right_key[0], left_key[1])
        return get_g_key(exchanged_left, exchanged_right)

    def key_to_g_parts(key):
        return ' '.join(key[0]), ' '.join(key[1])

    def get_product_key(prefix_terms):
        return tuple(sorted(prefix_terms))

    def parse_g_term(s):
        s = s.strip()
        sign = s[0] if s and s[0] in '+-' else '+'
        body = s[1:].strip() if s and s[0] in '+-' else s
        g_pos = body.index('g_{')
        prefix = body[:g_pos].strip()
        prefix_terms = prefix.split()
        magnitude = 1
        if prefix_terms:
            try:
                magnitude = float(prefix_terms[-1])
                if magnitude.is_integer():
                    magnitude = int(magnitude)
                prefix_terms = prefix_terms[:-1]
            except ValueError:
                magnitude = 1
        prefix = ' '.join(prefix_terms)
        prefix_key = get_product_key(prefix_terms)
        g_body = body[g_pos + len('g_{'):].split('}', 1)[0]
        left, right = [part.strip() for part in g_body.split(',', 1)]

        coef = magnitude if sign == '+' else -magnitude
        left_key = tuple(left.split())
        right_key = tuple(right.split())
        key = get_g_key(left_key, right_key)
        return prefix_key, prefix, key, coef, left, right

    def combine_exchange_terms(vals):
        entries = [[key, coef, left, right, False]
                   for key, (coef, left, right) in vals.items() if coef != 0]
        if not exchange:
            return entries

        combined = []
        used = set()
        entry_by_key = {entry[0]: i for i, entry in enumerate(entries)}
        for i, entry in enumerate(entries):
            if i in used:
                continue

            key, coef, left, right, _ = entry
            exchange_key = get_exchange_key(key)
            j = entry_by_key.get(exchange_key)
            if exchange_key is None or exchange_key == key or j is None or j in used:
                combined.append(entry)
                used.add(i)
                continue

            partner = entries[j]
            if coef + partner[1] != 0:
                combined.append(entry)
                used.add(i)
                continue

            base = entry if coef > 0 else partner
            direct_key, direct_coef = base[0], base[1]
            direct_left, direct_right = key_to_g_parts(direct_key)
            combined.append([direct_key, direct_coef, direct_left, direct_right, True])
            used.update({i, j})

        return combined

    def format_g_parts(coef, left, right, exchange_term=False):
        sign = '+' if coef > 0 else '-'
        magnitude = abs(coef)
        symbol = r'\tilde{g}' if exchange_term else 'g'
        coefficient = '' if magnitude == 1 else f'{magnitude} '
        integral = f'{symbol}_{{{left}, {right}}}'
        return sign, coefficient, integral

    def format_product(key, coef, left, right, exchange_term=False):
        sign, coefficient, integral = format_g_parts(coef, left, right, exchange_term)
        if key:
            return f'{sign} {coefficient}{key} {integral}'
        return f'{sign} {coefficient}{integral}'

    def format_g_term(coef, left, right, exchange_term=False):
        sign, coefficient, integral = format_g_parts(coef, left, right, exchange_term)
        return f'{sign} {coefficient}{integral}'

    def format_grouped_terms(key, entries):
        if len(entries) == 1:
            _, coef, left, right, exchange_term = entries[0]
            return format_product(key, coef, left, right, exchange_term) + '\n'

        terms = [format_g_term(coef, left, right, exchange_term)
                 for _, coef, left, right, exchange_term in entries]
        if key:
            return '+ ' + key + '( ' + ' '.join(terms) + ' )\n'
        return ''.join(term + '\n' for term in terms)

    strings = []
    strings_dict = {}
    for s in contracted_strings:
        if 'g_{' not in s:
            strings.append(s)
        else:
            prefix_key, prefix, key, coef, left, right = parse_g_term(s)
            if prefix_key not in strings_dict:
                strings_dict[prefix_key] = [prefix, {}]

            vals = strings_dict[prefix_key][1]
            if key not in vals:
                vals[key] = [0, left, right]
            vals[key][0] += coef

    for _, (prefix, vals) in strings_dict.items():
        entries = combine_exchange_terms(vals)
        if len(entries) == 0:
            continue
        strings.append(format_grouped_terms(prefix, entries))

    return strings


def plot_wick_diagram(operators, contractions, colors=None, width=None, end=''):
    r"""
    Plot Wick contraction diagram using graphviz.

    Args:
        operators (list): Operator strings.
        contractions (list): Contraction patterns.
        colors (list, optional): Colors for the contraction lines.
        width (float, optional): Line width for the contraction lines.
        end (str, optional): Symbols to append at the end of each line.
    """
    if isinstance(contractions[0], list): # loop over multiple contraction patterns
        return [plot_wick_diagram(operators, c, colors, width, end) for c in contractions]

    operators = get_list(operators)
    n_op = len(operators)

    orbs = [remove_dagger(op) for op in operators]
    creators = [is_creator(op) for op in operators]
    creators = [r'^{\dagger}' if c else '' for c in creators]

    if colors is None:
        colors = get_plot_colors(n_op//2)
    if width is None:
        width = .5 # in ex

    string = '\n'
    for k, (i1, i2, _, _) in enumerate(contractions):
        ops1 = ''
        if i1 > 0:
            for i in range(i1):
                ops1 += r'\hat{a}_{%s}%s' % (orbs[i], creators[i])
        ops2 = r'_{%s}%s' % (orbs[i1], creators[i1])
        for i in range(i1+1, i2):
            ops2 += r'\hat{a}_{%s}%s' % (orbs[i], creators[i])
        string += r'{\color{%s}\contraction[%2.1fex]{%s}{\hat{a}}{%s}{\hat{a}} }' % (colors[k], (width*(n_op//2-k)), ops1, ops2) + '\n'

    for i in range(n_op):
        string += r'\hat{a}_{%s}%s ' % (orbs[i], creators[i])
    string = string[:-1] + end + ' \\\\ \n'

    return string


def print_math(string, title, filename=None, latex=False):
    r"""
    Print mathematical expression in string format.

    Args:
        string (str): Mathematical expression.
        title (str): Title printed before the expression.
        filename (str, optional): If provided, save the expression to this file.
        latex (bool, optional): If True, print in LaTeX format.
    """
    if latex:
        string = string.replace('ell', r'\ell')
        string = string.replace('delta', r'\delta')
        string = string.replace('bar-sigma', r'{\bar{\sigma}}')
        string = string.replace('_', '_\\')
        string = string.replace('_\\{', r'_{')

    print(title)

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(string)
    else:
        print(string)
        print('')


def commutator(op1, op2, op3=None, sign='-'):
    r"""
    Compute the commutator [op1, op2] or double commutator
    [[op1, op2, op3] = ([[op1, op2], op3] + [op1, [op2, op3]]) / 2
    = (op1 op2 op3 + op3 op2 op1) - [[op1, op3]_+, op2]_+ / 2

    Args:
        op1: First operator string.
        op2: Second operator string.
        op3 (str, optional): Third operator string for a double commutator.
        sign (str, optional): Sign between the two terms in the commutator.

    Returns:
        tuple: ``(result, factor)`` for the commutator expansion.
    """
    if op3 is None:
        if isinstance(op1, list) and isinstance(op2, list):
            result = [[*op1, *op2], [*op2, *op1]]
        elif isinstance(op1, list):
            result = [[*op1, op2], [op2, *op1]]
        elif isinstance(op2, list):
            result = [[op1, *op2], [*op2, op1]]
        else:
            result = [[op1, op2], [op2, op1]]
        factor = [1, -1] if sign == '-' else [1, 1]

    else:
        result = [[op1, op2, op3], [op3, op2, op1],
                  [op1, op3, op2], [op3, op1, op2],
                  [op2, op1, op3], [op2, op3, op1]]
        factor = [1, 1] + [ -0.5 for _ in range(4)]
    return result, factor


def sqo_evaluation(bra, middle, ket, exceptions=[], title='', hamiltonian=None,
                   latex=True, diagram=False, colors=None, delta_style='comma',
                   symmetry=False, exchange=False):
    r"""
    Evaluate the Wick contractions for the given second-quantization operator (sqo) strings of bra, middle, and ket,
    while excluding specified operator pairs from contraction.

    Args:
        bra: Left excitation operator string.
        middle: Middle operator string.
        ket: Right excitation operator string.
        exceptions (list, optional): Operator pairs to exclude from contraction.
        title (str, optional): Title for the evaluation.
        hamiltonian (str, optional): Hamiltonian operator string to contract with the
            deltas. Uses ``middle`` if None.
        latex (bool, optional): If True, format the delta strings for LaTeX rendering.
        diagram (bool, optional): If True, plot the Wick contraction diagram.
        colors (list, optional): Colors for the contraction lines in the diagram.
        delta_style (str, optional): Style for formatting delta functions in the output.
        symmetry (bool, optional): If True, apply symmetry to the two-electron integrals in the output.
        exchange (bool, optional): If True, combine exchange-integral pairs as
            antisymmetrized two-electron integrals.

    Returns:
        tuple: ``(contractions, deltas, strings)``.
    """
    if isinstance(bra, list): # in order of left-to-right
        bra = ' '.join(bra)
    if isinstance(ket, list): # in order of left-to-right
        ket = ' '.join(ket)
    if hamiltonian is None: # take middle as hamiltonian by default
        hamiltonian = middle

    operators = bra + ' ' + middle + ' ' + ket
    print('operators:\n', operators)
    pairs = wick_pairs(operators, exceptions=exceptions, index=True)
    contractions = wick_contraction(operators, pairs, expand=True)

    print(title)
    if len(contractions) == 0:
        print('No valid Wick contraction patterns found.\n')
        return contractions, '', ''

    for i, pattern in enumerate(contractions):
        print('Pattern:', i+1)
        for (i1, i2, op1, op2) in pattern:
            print(f'Contracting {op1} with {op2};')
    print('')

    if diagram:
        strings = plot_wick_diagram(operators, contractions, end=';', colors=colors)
        print_math(' '.join(strings), 'Wick contraction diagram:\n', latex=latex)

    deltas = wick_delta(contractions, delta_style)

    strings = contract_hamil_delta(hamiltonian, deltas, symmetry, exchange)
    print_math(' '.join(strings), 'Contraction result:\n', latex=latex)

    return contractions, deltas, strings



if __name__ == '__main__':
    #operators = 'a_a^ i_b b_b† j_α c_b^ k_beta'
    operators = 's_alpha^ a_alpha p_sigma^ q_tau b_alpha^ t_alpha'
    exceptions = [('s_alpha^', 'a_alpha'), ('b_alpha^', 't_alpha')]
    pairs = wick_pairs(operators, exceptions=exceptions, index=True)
    print('pairs:', pairs)
    contractions = wick_contraction(operators, pairs, expand=True)
    print('contractions:', contractions)
    deltas = wick_delta(contractions, delta_style='upper_lower')
    print('deltas:', deltas)
    plot_wick_diagram(operators, contractions, end=';')

    h = 'p_sigma^ q_tau'
    contracted_strings = contract_hamil_delta(h, deltas)
    print('contracted_strings:', contracted_strings)
