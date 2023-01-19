from library.infrastructure.run_utils import ExperimentGrid, ExperimentManger


def cql_thunk(**kwargs):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cql_continuous import run_d4rl_cql
    run_d4rl_cql(**kwargs)


@ExperimentManger.register
def run_cql_gym_experiments():
    env_lists = []
    for agent in ['hopper', 'halfcheetah', 'ant', 'walker2d']:
        for dataset in ['medium-expert']:
            env_name = agent + '-' + dataset + '-v2'
            env_lists.append(env_name)

    grid = ExperimentGrid('cql_gym')
    grid.add(key='env_name', vals=env_lists, in_name=True)
    grid.add(key='seed', vals=[100, 200, 300], in_name=True)
    grid.add(key='min_q_weight', vals=[0.5, 2.0, 5.0], in_name=True, shorthand='min_q_weight')

    grid.run(cql_thunk, data_dir='results')


@ExperimentManger.register
def run_cql_ant_maze():
    env_lists = ['antmaze-umaze-v2']
    grid = ExperimentGrid('cql_gym')
    grid.add(key='env_name', vals=env_lists, in_name=True)
    grid.add(key='seed', vals=[100, 200, 300], in_name=True)
    grid.add(key='min_q_weight', vals=[0.1], in_name=True, shorthand='min_q_weight')
    grid.add(key='cql_threshold', vals=[-2.], in_name=True, shorthand='cql_threshold')

    grid.run(cql_thunk, data_dir='results')


if __name__ == '__main__':
    ExperimentManger.main()
