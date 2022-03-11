from __future__ import annotations

import random
from functools import reduce
from typing import Any, Callable

import aiger as A
import aiger_bv as BV
import aiger_dfa
import aiger_gridworld as GW
import aiger_ptltl as LTL
import attr
import funcy as fn
import networkx as nx
import numpy as np
from aiger_bdd import to_bdd, bdd_to_nx
from bidict import bidict
from dd import cudd
from dfa import DFA, dfa2dict, dict2dfa

from diss import DemoPrefixTree as PrefixTree
from diss.planners.tabular import TabularPolicy
from diss.domains.gridworld_circ import GridWorldCirc
from diss.concept_classes.dfa_concept import DFAConcept
from diss.experiment.concept_class import ignore_white, dont_count


BExpr = cudd.Function
Pos = tuple[int, int]
CODEC = {
    '.': 'white', 'r': 'red', 'b': 'blue',
    'r': 'red', 'g': 'green', 'y': 'yellow'
}
COLORS = frozenset(CODEC.values())


def bits_needed(n: int) -> int:
    return len(bin(n - 1)) - 2  # Number of bits needed to encode n.


def to_onehot(n: int, name: str):
    """Builds circuit mappins integers to onehot encodings.

    - Input is a word with enough bits to represent n as an unsigned int.
    - Output is 1 << min(x, n-1) where x in the integer interpretation of
      the input. If n is a power of two, this is simply 1 << x.
    """
    assert n > 1
    size = bits_needed(n)
    return BV.lookup(
        inlen=size,
        outlen=n,
        mapping={i: 1 << min(i, n - 1) for i in range(1 << size)},
        input=name,
        output=name,
        in_signed=False,
        out_signed=False,
    )


@attr.frozen
class GridWorldPlanner:
    start: tuple[int, int] | None
    gw: GridWorldCirc
    manager: cudd.BDD 
    horizon: int

    @staticmethod
    def from_string(
            buff: str,
            horizon: int,
            slip_prob: float=1/32,
            start: Pos | None = None,
            manager: cudd.BDD | None = None,
        ) -> GridWorldPlanner:
        gw = GridWorldCirc.from_string(
            buff=buff,
            start=(1, 1) if start is None else start,
            slip_prob=1/32,
            codec=CODEC,
        )

        if manager is None:
            manager = cudd.BDD()
            causal_order = []
            for i in range(bits_needed(gw.dim)):
                causal_order.append(f'x##time_0[{i}]')
            for i in range(bits_needed(gw.dim)):
                causal_order.append(f'y##time_0[{i}]')
            for t in range(horizon):
                causal_order.append(f'a##time_{t}[0]')
                causal_order.append(f'a##time_{t}[1]')
                causal_order.append(f'c##time_{t}[0]')
            manager.declare(*causal_order)
            manager.reorder({x: i for i, x in enumerate(causal_order)})
            manager.configure(reordering=False)

        return GridWorldPlanner(
            start=start,
            manager=manager,
            horizon=horizon,
            gw=gw
        )

    def to_demo(self, trc):
        start, *trc = trc

        demo = [ (None, 'env'), (start, 'ego')]
        for inputs in trc:
            demo.extend([ (inputs['a'], 'env'), (inputs['c'], 'ego')])
        return demo

    def lift_path(self, path, *, flattened=True):
        if flattened:
            dummy, start, *path = path
            assert dummy is None
            path = [{'a': a, 'c': c} for a, c in fn.chunks(2, path)]
        else:
            start, *path =  path

        aps = fn.pluck(0, self.gw.dyn_sense.simulate(path, latches={
            'x': 1 << (start[1] - 1), 'y': 1 << (start[0] - 1),  # Reversed for legacy reasons.
        }))
        aps = [fn.first(k for k, v in ap.items() if v == 1) for ap in aps]
        aps = ignore_white(aps)
        aps = dont_count(aps)
        return tuple(aps)

    def plan(self, concept, tree, psat, monolithic):
        dag = self.dfa2nx(concept)
        policy = LiftedPolicy.from_psat(dag, psat=psat, gw=self.gw)

        # Associcate each tree stree with a policy state.
        stack = [(tree.root, policy.root)]
        tree2policy = {}
        while stack:
            tstate, pstate = stack.pop()
            tree2policy[tstate] = pstate

            # Compute local mapping from dynamics transition to next pstate.
            for tstate2 in tree.tree.neighbors(tstate):
                action = tree.state(tstate2)  # tree states are next actions.
                pstate2 = policy.transition(pstate, action)
                stack.append((tstate2, pstate2))
        return CompressedMC(tree, policy, tree2policy, monolithic, self.lift_path)

    def dfa2nx(self, concept: DFAConcept) -> BExpr:
        bexpr = dag = bdd_to_nx(self.dfa2bdd(concept))

        for src, data in dag.nodes(data=True):
            label = data['label']
            if isinstance(label, bool):
                data['kind'] = label
            elif label.startswith('a'):
                data['kind'] = 'ego'
            else:
                data['kind'] = 'env'

        for src, tgt, data in dag.edges(data=True):
            if dag.nodes[src]['kind'] != 'env':
                continue
            label = dag.nodes[src]['label']
            if label.startswith('c'):
                bias = self.gw.slip_prob
            else:
                assert label.startswith('x') or label.startswith('y')
                bias = 0.5
            data['prob'] = 1 - bias if data['label'] else bias
        dag.graph['lvls'] = self.manager.var_levels
        return dag

    def dfa2bdd(self, concept: DFAConcept) -> BExpr:
        init = self.start is not None
        horizon = self.horizon
        monitor = self.dfa2monitor(concept)

        unrolled = monitor.aigbv \
                          .cone('SAT') \
                          .unroll(horizon, only_last_outputs=True, init=init)

        if not init:  # Hack to allow picking starting location.
            # Initialize DFA state.
            state = BV.decode_int(monitor.latch2init['state'], signed=False)
            unrolled <<= BV.uatom(monitor.aigbv.lmap['state'].size, state) \
                           .with_output('state##time_0')    \
                           .aigbv
            # Change encoding to make all assignments to initial (x, y) 
            # variables a valid start location.
            unrolled <<= to_onehot(8, 'x##time_0') | to_onehot(8, 'y##time_0')

        bexpr, *_ = to_bdd(
            unrolled, 
            manager=self.manager, 
            renamer=lambda _, x: x,
        )
        return bexpr

    def dfa2monitor(self, concept: DFAConcept) -> BV.AIGBV:
        # Add noop on white:
        def transition_with_noop(s, c):
            if c == 'white':
                return s
            return concept.dfa._transition(s, c)
        
        dfa = DFA(
            start=concept.dfa.start,
            outputs=concept.dfa.outputs,
            inputs=COLORS,
            label=concept.dfa._label,
            transition=transition_with_noop,
        )
        dfa = attr.evolve(
            concept.dfa,
            transition=transition_with_noop,
            inputs=COLORS,
            outputs=frozenset({True, False}),
        )
        circ, relabels, _ = aiger_dfa.dfa2aig(dfa)
        # Wrap circ i/o to interface with sensor.
        atoms = [BV.uatom(1, c) for c in COLORS]

        # Convert input.
        def get_idx(atom):
            c = fn.first(atom.inputs)
            return relabels['inputs'][c]['action'].index(True)

        atoms = sorted(atoms, key=get_idx)
        action = reduce(lambda x, y: x.concat(y), atoms).with_output('action')
        
        # Convert output.
        output = BV.uatom(2, 'output')
        for key, val in relabels['outputs'].items():
            if val is True:
                sat = output[key['output'].index(True)].with_output('SAT')
        monitor = action.aigbv >> circ >> sat.aigbv
        monitor = attr.evolve(monitor, aig=monitor.aig.lazy_aig)  # HACK: force lazy evaluation.
        return self.gw.dyn >> self.gw.sensor >> monitor


# =========================================================================#
#                 Lifted Policy + Compress Markov Chain                    #
# =========================================================================#

#               bdd-id    lvl   prev action  
# Node = tuple[  int  ,   int,  int | str]   # Lifted policy state.

def get_lvl(dag, node):
    label = dag.nodes[node]['label']
    if isinstance(label, bool):
        return len(dag.graph['lvls'])
    return dag.graph['lvls'][label]

def get_debt(dag, node1, node2):
    lvl1 = get_lvl(dag, node1)
    lvl2 = get_lvl(dag, node2)
    return lvl2 - lvl1 - 1

def walk(dag, curr, bits):
    for bit in bits:
        yield curr
        node, debt = curr
        if debt > 0:  # Don't care consumes bits.
            curr = (node, debt - 1)
            continue
        # Use bit for BDD transition.
        if dag.out_degree(node) == 0:
            break
        for kid in dag.neighbors(node):
            if bit == dag.edges[node, kid]['label']:
                break
        curr = (kid, get_debt(dag, node, kid))
    yield curr



@attr.frozen
class LiftedPolicy:
    policy: TabularPolicy
    gw: GridWorldCirc

    def psat(self, node = None): return self.policy.psat(node[0])
    def lsat(self, node = None): return self.policy.lsat(node[0])
        
    @property
    def root(self):
        dag, root = self.policy.dag, self.policy.root
        return (root, get_lvl(dag, root), None)

    @staticmethod
    def from_psat(unrolled, psat, gw, xtol=0.5):
        ctl = TabularPolicy.from_psat(unrolled, psat, xtol=xtol)
        return LiftedPolicy(ctl, gw)

    def prob(self, node, move, log = False):
        dag = self.policy.dag
        node1, debt1, _ = node 
        node2, debt2, action = move
        if not ((node1 != node2) or (debt1 > debt2 >= 0)):
            raise RuntimeError

        if isinstance(action, int):
            prob = 31 / 32 if action else 1/32
            return np.log(prob) if log else prob
        elif isinstance(action, tuple):
            return -np.log(2) if log else 0.5

        action = GW.dynamics.ACTIONS_C[action]
        bits = [action & 1, (action >> 1) & 1]
        curr = (node1, debt1)
        edges = fn.pairwise(walk(dag, (node1, debt1), bits))
        
        logp = 0
        for start, end in edges:
            if start[0] == end[0]:  # Don't care consumes bits.
                logp -= np.log(2)
            else:
                logp += self.policy.prob(start[0], end[0], log=True)

        assert end == (node2, debt2)
        return logp if log else np.exp(logp)

    def transition(self, pstate, action):
        dag = self.policy.dag
        if isinstance(action, str):  # action correspond to previous action.
            bits = GW.dynamics.ACTIONS_C[action]
            bits = [bits & 1, (bits >> 1) & 1]
        elif isinstance(action, tuple):
            y, x = action  # Flipped for legacy reasons.
            size = bits_needed(self.gw.dim)
            bits = list(BV.encode_int(size, x - 1, signed=False))
            bits.extend(BV.encode_int(size, y - 1, signed=False))
        else:
            bits = [action]
        node, debt = fn.last(walk(dag, pstate[:2], bits))  # QDD state.
        return (node, debt, action)

    def end_of_episode(self, pstate):
        node, debt, _ = pstate
        dag = self.policy.dag
        return (debt == 0) and (dag.out_degree(node) == 0)


# TODO: Change monolithic flag to generic subset filter.
@attr.frozen
class CompressedMC:
    """Compressed Markov Chain operating with actions."""
    tree: PrefixTree
    policy: LiftedPolicy
    tree2policy: dict[int, tuple[int, int]]
    monolithic: bool  # Monolithic experiment?
    lift_path: Callable[[Any], Any]  # TODO: annotate

    @property
    def edge_probs(self):
        edge_probs = {}
        for tree_edge in self.tree.tree.edges:
            dag_edge = [self.tree2policy[s] for s in tree_edge]
            edge_probs[tree_edge] = self.policy.prob(*dag_edge)
        return edge_probs
    
    def sample(self, pivot, win, attempts=20):
        # Sample until you give a path that respects subset properties.
        for i in range(attempts):
            result = self._sample(pivot, win)
            if result is None:
                return result
            word = self.lift_path(result[0])
            if self.monolithic:
                return result  # Allow violations of subset in sample.
            if (not win) and (('red' in word) or ('yellow' not in word)):
                continue
            return result

    def _sample(self, pivot, win):
        assert pivot > 0
        policy = self.policy
        state = self.tree2policy[pivot]

        if policy.psat(state) == float(not win):
            return None  # Impossible to realize is_sat label.

        sample_lprob: float = 0  # Causally conditioned logprob.
        path = list(self.tree.prefix(pivot))
        if policy.end_of_episode(state):
             moves = []
        else:
            prev_ego = isinstance(state[-1], str)

            # Make sure to deviate from prefix tree at pivot.
            actions = {0, 1} if prev_ego else set(GW.dynamics.ACTIONS_C)
            actions -= {self.tree2policy[s][-1] for s in self.tree.tree.neighbors(pivot)}

            tmp = {policy.transition(state, a) for a in actions}

            moves = list(m for m in tmp if policy.psat(m) != float(not win))

        if not moves:
            return None  # Couldn't deviate
        
        # Sample suffix to path conditioned on win.
        while moves:
            # Apply bayes rule to get Pr(s' | is_sat, s).
            priors = np.array([policy.prob(state, m) for m in moves])
            likelihoods = np.array([policy.psat(m) for m in moves])
            normalizer = policy.psat(state)

            if not win:
                likelihoods = 1 - likelihoods
                normalizer = 1 - normalizer

            probs =  priors * likelihoods / normalizer
            prob, state = random.choices(list(zip(probs, moves)), probs)[0]
            if policy.policy.dag.nodes[state[0]]['kind'] == 'ego':
                sample_lprob += np.log(prob)

            # Note: win/lose are strings so the below still works...
            action = state[-1]
            path.append(action)

            if policy.end_of_episode(state):
                moves = []
            else:
                prev_ego = isinstance(action, str)
                actions = {0, 1} if prev_ego else set(GW.dynamics.ACTIONS_C)
                moves = [policy.transition(state, a) for a in actions]

        return path, sample_lprob
