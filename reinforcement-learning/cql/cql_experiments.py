from library.infrastructure.run_utils import ExperimentGrid, ExperimentManger


def cql_thunk(**kwargs):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cql_continuous import run_d4rl_cql
    run_d4rl_cql(**kwargs)


@ExperimentManger.register
def run_cql_gym_experiments():
    env_lists = ['walker2d-medium-v2']
    grid = ExperimentGrid('cql_gym')
    grid.add(key='env_name', vals=env_lists, in_name=True)
    grid.add(key='seed', vals=[100, 200, 300], in_name=True)
    grid.add(key='min_q_weight', vals=[0.2], in_name=True, shorthand='min_q_weight')

    grid.run(cql_thunk, data_dir='results')


if __name__ == '__main__':
    ExperimentManger.main()
