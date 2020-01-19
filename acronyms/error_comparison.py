import json
import os

import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Main script for Acronym Training Model')

    # Functional Arguments
    parser.add_argument('--e1', default='baseline_mimic')
    parser.add_argument('--e2', default='lmc-context-section_mimic')

    args = parser.parse_args()

    e1_fn = os.path.join('weights', args.e1, 'results', 'error_tracker.json')
    e2_fn = os.path.join('weights', args.e2, 'results', 'error_tracker.json')

    with open(e1_fn, 'r') as fd:
        e1_analysis = json.load(fd)
    with open(e2_fn, 'r') as fd:
        e2_analysis = json.load(fd)

    e1_correct = set(e1_analysis['correct'])
    e1_errors = set(e1_analysis['error'])

    e2_correct = set(e2_analysis['correct'])
    e2_errors = set(e2_analysis['error'])

    N = len(e1_correct) + len(e1_errors)
    assert len(e1_correct) + len(e1_errors) == len(e2_correct) + len(e2_errors)

    both_correct = e1_correct.intersection(e2_correct)
    both_wrong = e1_errors.intersection(e2_errors)

    e1_wrong = e1_errors - e2_errors
    e2_wrong = e2_errors - e1_errors

    e1_right = e1_correct - e2_correct
    e2_right = e2_correct - e1_correct

    assert e1_right == e2_wrong
    assert e1_wrong == e2_right

    print('N={}'.format(N))
    print('{} --> Correct={}, Incorrect={}'.format(args.e1, len(e1_correct), len(e1_errors)))
    print('{} --> Correct={}, Incorrect={}'.format(args.e2, len(e2_correct), len(e2_errors)))

    confusion_mat = np.zeros([2, 2])

    confusion_mat[0, 0] = len(both_wrong)
    confusion_mat[1, 1] = len(both_correct)
    confusion_mat[0, 1] = len(e2_right)
    confusion_mat[1, 0] = len(e1_right)

    print('Error Confusion Matrix: Left={}, Top={}'.format(args.e1, args.e2))
    print(confusion_mat)

    out_dir = os.path.join('weights', '{}-{}'.format(args.e1, args.e2))
    if not os.path.exists(out_dir):
        print('Making dir {}'.format(out_dir))
        os.mkdir(out_dir)

    out_fn = os.path.join(out_dir, 'comparison.json')
    print('Saving results to {}'.format(out_fn))
    with open(out_fn, 'w') as fd:
        json.dump({
            '{}_correct'.format(args.e1): list(e1_right),
            '{}_correct'.format(args.e2): list(e2_right),
        }, fd)

    e1_right_sample = np.random.choice(list(e1_right), 10)
    e2_right_sample = np.random.choice(list(e2_right), 10)
    print('Only {} correct sample --> {}'.format(args.e1, e1_right_sample))
    print('Only {} correct sample --> {}'.format(args.e2, e2_right_sample))
