import pandas as pd
import numpy as np

def get_errors(file_path, target_FEs):
    data = pd.read_csv(file_path, index_col = False)
    errors = []
    for fe in target_FEs:
        df = data.loc[ data.FEs > fe, ['FEs', 'error'] ].min()
        if pd.isnull(df.error):
            df = data.iloc[-1]
        errors.append( [df.FEs, df.error] )
    return np.array(errors)


def generate_table(problem):
    target_FEs = [ 5e2, 1e3, 1e4, 2e4 ]
    for repeat in range(1, 26):
        file_path = '%s_%d.csv' % (problem, repeat)
        errors = get_errors(file_path, target_FEs)
        print(file_path)
        print(errors)
        input()

def main():
    data_path = 'data/2017-05-22_ori_bandit'
    for algo in ['CMA', 'PSO', 'ACOR']:
        for function_id in range(1, 26):
            for dimension in [2, 10, 30, 50]:
                problem = '%s/%s/F%d_%dD' % \
                          (data_path, algo, function_id, dimension)
                generate_table(problem)


if __name__ == '__main__':
    main()
