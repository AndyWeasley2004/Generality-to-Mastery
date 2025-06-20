import os
import pickle5 as pickle
from sklearn.model_selection import train_test_split

def split_events(data_root, output_dir, dataset_name):
    os.makedirs(output_dir, exist_ok=True)

    pkl_files = os.listdir(data_root)
    # dataset_name = 'maestro'
    train_set, valid_set = train_test_split(pkl_files, test_size=0.05, random_state=42)
    pickle.dump(train_set, open(os.path.join(output_dir, f'{dataset_name}_train.pkl'), 'wb'))
    pickle.dump(valid_set, open(os.path.join(output_dir, f'{dataset_name}_valid.pkl'), 'wb'))

    print(' > num files: ', len(train_set) + len(valid_set))
    print(' > train, valid:', len(train_set), len(valid_set))


if __name__ == '__main__':
    # score_data/performance_data
    dataset_name = 'top5_subset'
    data_root = f'../performance_data/full/{dataset_name}/events'
    output_dir = '../performance_data/splits/'
    split_events(data_root, output_dir, dataset_name)
