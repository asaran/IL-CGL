import argparse
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ##################################################
    # ##             Algorithm parameters             ##
    # ##################################################

    parser.add_argument("--all_games", action="store_true", default=False)
    parser.add_argument("--dir", type=str, default='')

    parser.add_argument("--ckpt", type=int, default=43000)
    parser.add_argument("--weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)


    args = parser.parse_args()

    if args.all_games:
        env_names = ['alien','asterix','bank_heist','berzerk','breakout','centipede','demon_attack','enduro','freeway','frostbite','hero','montezuma_revenge','mspacman','name_this_game','phoenix','riverraid','road_runner','seaquest','space_invaders','venture']
    else:
        env_names = ['asterix','breakout','centipede','phoenix','mspacman','seaquest']

    avgs, stds = [], []

    files = os.listdir(args.dir)
    for env in env_names:
        empty = False

        file_name = args.dir.split('/') 
        file_name = "_".join(file_name)
        result_file = 'eval/'+env+file_name+env+'_KL_'+str(args.weight)+'_seed'+str(args.seed)+'_checkpoints_'+str(args.ckpt)+'_evaluation.txt'

        if not os.path.isfile(result_file):
            print('File does not exist: ', result_file)
            empty = True
            avgs.append('#')
            stds.append('#')
            continue

        f = open(result_file, "r")
        scores, std_error = [], 0.0
        for rollout in f:
            rollout = rollout.strip('\n')
            scores.append(float(rollout))

        avg_score = np.mean(scores)
        avgs.append(avg_score)

        std_error = np.std(scores)/np.sqrt(len(scores))
        stds.append(std_error)


    # print avg scores
    print('Average Scores*****')
    for a,s in zip(avgs,stds):
        if a!='#':
            print("%.2f +/- %.2f" % (a,s))
        else:
            print('#')
    print('\n\n')

