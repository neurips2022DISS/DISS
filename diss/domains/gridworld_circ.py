from __future__ import annotations
from collections import defaultdict

import aiger as A
import aiger_bv as BV
import aiger_gridworld as GW
import attr
import funcy as fn


@attr.frozen
class GridWorldCirc:
    dyn: BV.AIGBV 
    sensor: BV.AIGBV

    @staticmethod
    def from_string(buff, start, codec, slip_prob=1/32) -> GridWorldCirc:
        rows = buff.split()
        widths = {len(row) for row in rows}
        assert len(widths) == 1
        dim = fn.first(widths)
        assert len(rows) == dim

        # Create gridworld circuit.
        dyn = GW.gridworld(
            dim,
            start=(start[0], dim + 1 - start[1]),  # Force origin to top left.
            compressed_inputs=True
        )
        slip = BV.uatom(1, 'c').repeat(2) & BV.uatom(2, 'a')
        slip = slip.with_output('a').aigbv
        dyn <<= slip

        # Create sensor to observe aps.
        STATE = BV.uatom(2 * dim, 'state')
        
        ## 1. Setup individual predicates for codec.
        overlay = defaultdict(lambda: BV.uatom(1, 0))
        for y, row in enumerate(rows):
            aps = (codec.get(s, 'white') for s in row)
            for x, ap in enumerate(aps):
                state = (1 << x) | (1 << ((dim - y - 1) + dim))
                overlay[ap] |= STATE == state

        ## 2. Combine predicates into a single sensor.
        sensor = BV.aig2aigbv(A.empty())
        for name, ap in overlay.items():
            sensor |= ap.with_output(name).aigbv

        return GridWorldCirc(dyn, sensor)
