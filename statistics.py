import os
import pandas as pd
import numpy as np

def get_errors(file_path, target_FEs):
    try:
        data = pd.read_csv(file_path, index_col = False)
    except FileNotFoundError as e:
        print(e)
        errors = np.empty((len(target_FEs),2))
        errors[:] = np.NAN
        return errors

    errors = []
    for fe in target_FEs:
        df = data.loc[ data.FEs > fe, ['FEs', 'error'] ].min()
        if pd.isnull(df.error):
            df = data.iloc[-1]
        errors.append( [df.FEs, df.error] )
    return np.array(errors)


def generate_table(data_path, problem, csv_file=None):
    repeats = range(1, 26)
    target_FEs = [ 5e2, 1e3, 1e4, 2e4 ]
    all_errors = np.empty((len(repeats), len(target_FEs)))
    all_errors[:] = np.NAN
    
    for i, repeat in enumerate(repeats):
        file_path = '%s/%s_%d.csv' % (data_path, problem, repeat)
        errors = get_errors(file_path, target_FEs)
        #print(file_path)
        all_errors[i] = errors[:,1]

    all_errors.sort(axis=0)
    mean = np.mean(all_errors, axis=0)
    std  = np.std(all_errors, axis=0)
    data = np.concatenate((all_errors[[ 0, 6, 12, 18, 24]], mean[None,:], std[None,:]))

    columns = [ 'FEs = %.0e' % f for f in target_FEs ]
    index = ['1st(Best)', '7th', '13th(Median)', '19th', '25th(Worst)', 'Mean', 'Std']
    df = pd.DataFrame( data, columns=columns, index=index )

    if csv_file is not None:
        print('\nSaving to %s/results/%s.csv...' % (data_path, problem))
        print(df)
        df.to_csv( csv_file )

    return df



def main():
    data_dir = 'data/2017-05-22_original_algos'
    for algo in ['CMA', 'PSO', 'ACOR']:
        #for dimension in [2, 10, 30, 50]:
        for dimension in [2]:
            for function_id in range(1, 26):
                data_path = '%s/%s' % (data_dir, algo)
                problem = 'F%d_%dD' % (function_id, dimension)
                #generate_table(data_path, problem)

                #'''
                if not os.path.exists( '%s/results' % data_path ):
                    os.makedirs( '%s/results' % data_path )
                csv_file = '%s/results/%s.csv' % (data_path, problem)
                generate_table(data_path, problem, csv_file = csv_file)
                #'''

if __name__ == '__main__':
    main()
