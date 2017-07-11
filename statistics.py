import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        if any(data.FEs > fe):
            df = data[ data.FEs > fe ].iloc[0]
        else:
            df = data.iloc[-1]
        FEs, error = df.FEs, df.error
        errors.append( [FEs, error] )
    return np.array(errors)


def generate_table(data_path, problem, target_FEs):
    repeats = range(1, 26)
    all_errors = np.empty((len(repeats), len(target_FEs)))
    all_errors[:] = np.NAN
    
    for i, repeat in enumerate(repeats):
        file_path = '%s/%s_%d.csv' % (data_path, problem, repeat)
        errors = get_errors(file_path, target_FEs)
        #print(file_path)
        all_errors[i] = errors[:,1]

    all_errors.sort(axis=0)
    #mean = np.mean(all_errors, axis=0)
    #std  = np.std(all_errors, axis=0)
    mean = np.nanmean(all_errors, axis=0)
    std  = np.nanstd(all_errors, axis=0)
    data = np.concatenate((all_errors[[ 0, 6, 12, 18, 24]], mean[None,:], std[None,:])) 
    columns = [ 'FEs = %.0e' % f for f in target_FEs ]
    index = ['1st(Best)', '7th', '13th(Median)', '19th', '25th(Worst)', 'Mean', 'Std']
    df = pd.DataFrame( data, columns=columns, index=index )

    return df



def plot_error():
    data_dir = 'data/2017-05-22_ori_bandit'
    target_FEs = [ 5e2, 1e3, 5e3, 1e4, 2e4 ]
    #for dimension in [2, 10, 30, 50]:
    for dimension in [2]:
        for function_id in range(1, 26):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale( "log", nonposx='clip' )
            ax.set_yscale( "log", nonposy='clip' )
            #ax.set_ylim(1e-9)

            for algo in ['CMA', 'PSO', 'ACOR']:

                data_dir = 'data/2017-05-22_original_algos'
                data_path = '%s/%s' % (data_dir, algo)
                problem = 'F%d_%dD' % (function_id, dimension)
                df = generate_table(data_path, problem, target_FEs)
                print('\n',algo, problem)
                print(df)

                X = np.array(target_FEs)
                Y = np.array(df.loc['Mean'].values)
                Y_error = np.array(df.loc['Std'].values)
                ax.plot( X, Y, '-o', label=algo )
                #ax.errorbar( X, Y, Y_error, fmt='o', label=algo )

                data_dir = 'data/2017-05-22_ori_bandit'
                data_path = '%s/%s' % (data_dir, algo)
                problem = 'F%d_%dD' % (function_id, dimension)
                df = generate_table(data_path, problem, target_FEs)
                print(df)
                '''
                if not os.path.exists( '%s/results' % data_path ):
                    os.makedirs( '%s/results' % data_path )
                csv_file = '%s/results/%s.csv' % (data_path, problem)
                print('\nSaving to %s/results/%s.csv...' % (data_path, problem))
                print(df)
                df.to_csv( csv_file )
                '''


                X = np.array(target_FEs)
                Y = np.array(df.loc['Mean'].values)
                Y_error = np.array(df.loc['Std'].values)
                ax.plot( X, Y, '-o', label='Bandit+%s'%algo )
                #ax.errorbar( X, Y, Y_error, fmt='o', label='Bandit+%s'%algo )

            plt.legend()
            plt.title( problem )
            plt.xlabel('FEs')
            plt.ylabel('Error')
            plt.savefig( 'plots/2017-05-23_ori_bandit/%s.png' % problem )
            plt.close(fig)

def main():
    path = 'data'
    #orig_dir = '2017-05-22_original_algos'
    orig_dir = '2017-07-09_original'
    data_dir = '2017-07-09_optimize_bandit'
    target_FEs = [ 5e2, 1e3, 5e3, 1e4, 2e4 ]
    #for dimension in [2, 10, 30, 50]:
    for dimension in [2]:
        for function_id in range(1, 26):
            #function_id = 5
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xscale( "log", nonposx='clip' )
            ax.set_yscale( "log", nonposy='clip' )
            #ax.set_ylim(1e-9)

            for algo in ['CMA', 'PSO', 'ACOR']:
            #for algo in ['CMA', 'ACOR']:

                data_path = '%s/%s/%s' % (path, orig_dir, algo)
                problem = 'F%d_%dD' % (function_id, dimension)
                df = generate_table(data_path, problem, target_FEs)
                print('\n',algo, problem)
                print(df)

                X = np.array(target_FEs)
                Y = np.array(df.loc['Mean'].values)
                Y_error = np.array(df.loc['Std'].values)
                ax.plot( X, Y, '-o', label=algo )
                #ax.errorbar( X, Y, Y_error, fmt='-o', label=algo )

                if algo == 'PSO':
                    continue
                data_path = '%s/%s/%s' % (path, data_dir, algo)
                problem = 'F%d_%dD' % (function_id, dimension)
                df = generate_table(data_path, problem, target_FEs)
                print(df)
                '''
                if not os.path.exists( '%s/results' % data_path ):
                    os.makedirs( '%s/results' % data_path )
                csv_file = '%s/results/%s.csv' % (data_path, problem)
                print('\nSaving to %s/results/%s.csv...' % (data_path, problem))
                print(df)
                df.to_csv( csv_file )
                '''


                X = np.array(target_FEs)
                Y = np.array(df.loc['Mean'].values)
                Y_error = np.array(df.loc['Std'].values)
                ax.plot( X, Y, '-o', label='Bandit+%s'%algo )
                #ax.errorbar( X, Y, Y_error, fmt='-o', label='Bandit+%s'%algo )

            plt.legend()
            plt.title( problem )
            plt.xlabel('FEs')
            plt.ylabel('Error')
            if not os.path.exists( 'plots/%s' % data_dir ):
                os.makedirs( 'plots/%s' % data_dir )
            plt.savefig( 'plots/%s/%s.png' % (data_dir,problem) )
            plt.close(fig)
            #input()


if __name__ == '__main__':
    main()
