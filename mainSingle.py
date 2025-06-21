import argparse
import pickle
import time
from util import Data, split_validation, init_seed
from modelSingle import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='movielens', help='dataset name: Instacart/lastfm/Cell_Phones_and_Accessories/Sports_and_Outdoors/Home_and_Kitchen/sample')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size 100')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--layer', type=float, default=6, help='the number of self attention layer')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_item_negatives', type=int, default=100, help='the number of item negatives 100')
parser.add_argument('--num_sess_negatives', type=int, default=100, help='the number of session negatives 100')
parser.add_argument('--alpha', type=float, default=0.01, help='item ssl task maginitude 0.01')
parser.add_argument('--beta', type=float, default=0.01, help='session ssl task maginitude 0.01')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(0)

def main():
    # list[0]:session list[1]:label
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    # init_seed(2023, True)

    if opt.dataset == 'Instacart':
        n_node = 12616
    elif opt.dataset == 'InstacartSpa':
        n_node = 21432
    else:
        print("unkonwn dataset")
    # data_formate: sessions, price_seq, matrix_session_item matrix_pv
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=True, n_node=n_node)
    model = trans_to_cuda(SemanticContrast(n_node=n_node, lr=opt.lr, layers=opt.layer, l2=opt.l2, alpha=opt.alpha, beta=opt.beta, dataset=opt.dataset, num_heads=opt.num_heads, emb_size=opt.embSize, batch_size=opt.batchSize, num_item_negatives=opt.num_item_negatives,num_sess_negatives=opt.num_sess_negatives))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        print('Epoch: %d\ttrain_loss:\t%.3f\tPrec@20: %.3f\tMRR@20: %.3f\tNDCG20: %.3f' %
              (epoch, total_loss, metrics['hit20'], metrics['mrr20'], metrics['ndcg20']))
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    main()
