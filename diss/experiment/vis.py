import dfa
import pydot
from diss.concept_classes.dfa_concept import remove_stutter, DFAConcept
from collections import defaultdict


COLOR_ALIAS = {
    'white': 'white',
    'yellow': '#ffff00', 
    #'brown': '#ffb081',
    'red': '#ff8b8b',
    'blue': '#afafff', 
    'green' : '#8ff45d'
}


# adapted from the dfa library
def get_dot(dfa_):
    dfa_dict, init = dfa.dfa2dict(dfa_)
    remove_stutter(dfa_dict)
    g = pydot.Dot(rankdir="LR")

    nodes = {}
    for i, (k, (v, _)) in enumerate(dfa_dict.items()):
        shape = "doublecircle" if v else "circle"
        nodes[k] = pydot.Node(i+1, label=f"{k}", shape=shape, color="white", fontcolor="white")
        g.add_node(nodes[k])

    edges = defaultdict(list)
    for start, (_, transitions) in dfa_dict.items():        
        for action, end in transitions.items():
            color = COLOR_ALIAS[str(action)]
            edges[start, end].append(color)
    
    init_node = pydot.Node(0, shape="point", label="", color="white")
    g.add_node(init_node)
    g.add_edge(pydot.Edge(init_node, nodes[init], color="white"))

    for (start, end), colors in edges.items():
        #color_list = f':'.join(colors)
        #g.add_edge(pydot.Edge(nodes[start], nodes[end], color=color_list))
        for color in colors:
            g.add_edge(pydot.Edge(nodes[start], nodes[end], label='◼', fontcolor=color, color="white"))
    g.set_bgcolor("#002b36")        
    return g


def view_dfa(dfa_or_concept):
    from IPython.display import SVG
    if isinstance(dfa_or_concept, DFAConcept):
        dfa_or_concept = dfa_or_concept.dfa
    pdot = get_dot(dfa_or_concept)
    display(SVG(pdot.create_svg()))
