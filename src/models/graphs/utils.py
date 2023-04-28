from typing import Union

import graphviz

from search_space_units import CellSpecification


def save_cell_dag_image(cell_spec: Union[str, CellSpecification], save_path: str):
    # parse cell if provided in str format
    if isinstance(cell_spec, str):
        cell_spec = CellSpecification.from_str(cell_spec)

    block_count = len(cell_spec)

    g = graphviz.Digraph(filename='cell_graph.gz', directory=save_path,
                         graph_attr=dict(rankdir='LR', ordering='out', splines='true'),
                         edge_attr=dict(arrowhead='vee'))

    # generate lookback nodes
    for i in cell_spec.used_lookbacks:
        g.node(f'lb{i}', f'{i}', shape='circle', color='red')

    # generate operator nodes
    for i, op in enumerate(cell_spec.operators):
        g.node(f'op{i}', op, shape='rect', color='blue')

    # generate "block join op" nodes and connect the two related operator nodes to it
    for i in range(block_count):
        g.node(f'add{i}', f'add{i}', shape='diamond', color='green')
        g.edges([(f'op{i * 2}', f'add{i}'), (f'op{i * 2 + 1}', f'add{i}')])

    # connect the inputs (lookbacks and possibly internal blocks) to the operators using them
    for i, inp in enumerate(cell_spec.inputs):
        # lookback
        if inp < 0:
            # NOTE: ':s' is compass direction 'sud', improving edge head placement
            g.edge(tail_name=f'lb{inp}', head_name=f'op{i}:w')
        # input from another block
        else:
            g.edge(tail_name=f'add{inp}', head_name=f'op{i}:w')

    # add output node and connect the unused block to it (adding concat when there are multiple ones)
    g.node('out', 'out', shape='circle', color='red')
    if len(cell_spec.unused_blocks) > 1:
        g.node(f'concat', shape='rect')
        g.edges([(f'add{i}', 'concat') for i in cell_spec.unused_blocks])
        g.edge(tail_name='concat', head_name='out:w')
    else:
        g.edge(tail_name=f'add{block_count - 1}', head_name='out:w')

    g.render()
