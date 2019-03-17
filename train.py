import numpy as np
import pandas as pd
import argparse
import math
import csv
import pickle

parser = argparse.ArgumentParser(description='neural network for recognizing hand written alphabets')
parser.add_argument( '--pretrain',type=bool,metavar='', help='(pretrain)')
parser.add_argument( '--testing',type=bool,metavar='', help='(testing)')
parser.add_argument( '--state',type=int,metavar='', help='(state)')
parser.add_argument( '--lr', type=float, metavar='',help='(initial learning rate for gradient descent based algorithms)')
parser.add_argument('--momentum', type=float, metavar='',help='(momentum to be used by momentum based algorithms)')
parser.add_argument( '--num_hidden', type= int,metavar='', help='(number of hidden layers excluding i/p and o/p layer)')
parser.add_argument( '--sizes',type=list,metavar='', help='(a comma separated list for the size of each hidden layer)')
parser.add_argument('--activation', type=str, metavar='',help='(the choice of activation function - valid values are tanh/sigmoid)')
parser.add_argument('--loss', type=str,metavar='', help='(possible choices are squared error[sq] or cross entropy loss[ce])')
parser.add_argument('--opt',type=str,metavar='', help='(the optimization algorithm to be used: gd, momentum, nag, adam')
parser.add_argument('--batch_size',type=int,metavar='', help='(the batch size to be used)')
parser.add_argument('--epochs',type=int,metavar='', help='(number of passes over the data)')
parser.add_argument('--anneal',type=bool,metavar='', help='(if true the algorithm should halve the learning rate if at any epoch the validation loss decreases and then restart that epoch)')
parser.add_argument( '--save_dir',type=str, metavar='',help='(the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network)')
parser.add_argument('--expt_dir', type=str,metavar='', help='(the directory in which the log files will be saved )')
parser.add_argument('--train', type=str,metavar='', help='(path to the Training dataset)')
parser.add_argument('--val', type=str, metavar='',help='(path to the validation dataset)')
parser.add_argument('--test', type=str, metavar='',help='(path to the Test dataset)')

args = parser.parse_args()



def save_weights(list_of_weights, epoch):
      with open(args.save_dir +'weights_{}.pkl'.format(epoch), 'wb') as f:
             pickle.dump(list_of_weights, f)

# def save_weights(list_of_weights, epoch):
#      with open(args.save_dir +'weights_{}.pkl'.format(epoch), 'w') as f:
#             thewriter = csv.writer(f)
#             #row, col = y_predict.shape
#             thewriter.writerow(weight_matrix_list)
#             # rowlen = len(weight_matrix_list)
#             # for i in range(rowlen):
#             #     thewriter.writerow(weight_matrix_list[i])

def load_weights(state):
      with open(save_dir +'weights_{}.pkl'.format(state)) as f:
              list_of_weights = pickle.load(f)
      return list_of_weights

def encode_into_prob(x,max_val):
    len=x.size
    y_list=[]
    for i in range(len):
        temp=[]
        for j in range(max_val):
            if(x[i]==j):
                temp.append(1)
            else:
                temp.append(0)
            #print(temp)
        y_list.append(temp)
    return y_list


def feedforward(batch_train, hidden_layers):
    global weight_matrix_list
    global bias_list
    global preactivation_list
    preactivation_list = []
    global activation_list
    activation_list = []
    global y_hat
    y_hat = []
    a_new = []
    a_last_layer = []
    nos_data = len(batch_train)
    # print(nos_data)
    h = batch_train.transpose()
    preactivation_list.append(h)
    activation_list.append(h)

    for j in range(hidden_layers):

        a_new = bias_list[j] + np.matmul(weight_matrix_list[j].transpose(), activation_list[j])
        preactivation_list.append(a_new)

        h = 1.0 / (1.0 + np.exp(-1 * preactivation_list[j + 1]))
        activation_list.append(h)

    j = hidden_layers
    a_last_layer = bias_list[j] + np.matmul(weight_matrix_list[j].transpose(), activation_list[j])
    preactivation_list.append(a_last_layer)

    y_hat = np.array(output(a_last_layer, batch_train))
    y_hat = y_hat.transpose()

def output(a_last_layer, batch_train):
    y_hat_initial = []
    y_hat_local = []
    nos_data = len(batch_train)
    for i in range(nos_data):
        y_hat_initial.append(softmax(a_last_layer[:, i:i + 1].transpose()))

    length = len(y_hat_initial)
    for i in range(length):
        y_hat_local.append(y_hat_initial[i][0])

    return y_hat_local


def softmax(row_vector):
    sum_all = 0
    row_vector = np.exp(row_vector)
    sum_all = np.sum(row_vector, axis=1)  # print("row_vector", row_vector)
    row_vector = (row_vector * 1.0 / sum_all)  # print("sum_all",sum_all)
    return row_vector


def backpropagation(y_true_batch):

    global grad_a_k_list
    grad_a_k_list = []

    global grad_w_k_list
    grad_w_k_list = []

    global grad_h_k_list
    grad_h_k_list = []

    global grad_b_k_list
    grad_b_k_list = []

    global grad_b
    grad_b = []

    global grad_g_dash_a_k_list
    grad_g_dash_a_k_list = []

    global grad_a_last
    grad_a_last = []

    k = num_hidden + 1


    grad_g_dash_a_k_list = g_dash(preactivation_list)

    grad_a_last = -np.subtract(y_true_batch, y_hat)

    grad_a_k_list.insert(0, grad_a_last)

    while (k > 0):

        grad_w_k_list.insert(0, np.matmul(grad_a_k_list[0], preactivation_list[k - 1].transpose()))

        grad_b_k_list.insert(0, grad_a_k_list[0])

        grad_h_k_list.insert(0, np.matmul(grad_w_k_list[0].transpose(), grad_a_k_list[0]))

        grad_a_k_list.insert(0, np.multiply(grad_h_k_list[0], grad_g_dash_a_k_list[k - 1]))

        k = k - 1
    #         print("grad_a_ik_list",grad_a_k_list[0])

    sum_all_grad_b_of_data(grad_b_k_list)


def sum_all_grad_b_of_data(grad_b_k_list):
    length = len(grad_b_k_list)
    i = 0
    while (i < length):
        row = []
        row = np.sum(grad_b_k_list[i], 1)
        j = len(row)
        row = row[:, np.newaxis]
        grad_b.append(row)
        i += 1


def g_dash(preactivation_list):
    length = len(preactivation_list)
    a_modified = preactivation_list[0:length]
    g_dash_a_k_list = []
    for i in range(length):
        a = (1.0 / (1 + np.exp(-a_modified[i]))) * (1 - (1.0 / (1 + np.exp(-a_modified[i]))))
        g_dash_a_k_list.append(a)
    return g_dash_a_k_list

def decode_yhat_to_classes(y_hat_local):
    col = len(y_hat_local[0])
    max_val=0
    for j in range(col):
        max_val = y_hat_local[0][j]
        index=0
        for i in range(9):
            if(y_hat_local[i+1][j] > max_val):
                y_hat_local[index][j]=0
                index = i+1
                max_val=y_hat[i+1][j]
            else :
                y_hat_local[i+1][j]=0
        y_hat_local[index][j]=1
    return y_hat_local


def classifcation(y_hat):
    y_predict=[]
    col = len(y_hat[0])
    for j in range(col):
        for i in range(10):
            if(y_hat[i][j]==1):
                y_predict.append(i+1)
                break
    return np.array(y_predict)


def cal_loss(y_hat, y_true):
    loss = 0
    for i in range(10):
        if(y_hat[i]!=0):
            loss += y_true[i] * math.log10(y_hat[i])

    return loss


def cal_error(y_hat, y_true):
    true = classifcation(y_true)
    predicted = classifcation(decode_yhat_to_classes(y_hat))
    if (true != predicted):
        return 1
    return 0

def add_id_to_y_predict(y_predict):
    ide=[]
    row=len(y_predict)
    for i in range(row):
        ide.append(i)
    ide=np.array(ide)
    ide=ide[:,np.newaxis]
    y= np.hstack((ide,y_predict))
    return y

def grad_descent(weight_matrix_list, bias_list, xtrain, num_hidden, sizes, eta, batch_size,epochs,expt_dir, differ_train_val, y_true):
    len_w = len(weight_matrix_list)
    len_b = len(bias_list)
    data_length = len(xtrain)
    no_of_batch = int(data_length / batch_size)
    steps = 0
    loss = 0
    error = 0
    nos_points = 0
    if (differ_train_val == 1):
        file = open(str(expt_dir) + "log_train.txt", 'w')
        #print(differ_train_val, file)
    else :
        file1 = open(str(expt_dir) + "log_val.txt", 'w')
        #print(differ_train_val, file1)
    for t in range(epochs):
        save_weights((weight_matrix_list), (epochs))
        for batch in range(no_of_batch):
            dw = []
            db = []
            for i in range(int(num_hidden) + 1):
                dw.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
                db.append(0 * np.random.rand(sizes[i + 1], 1))

            for data_index_in_batch in range(batch_size):
                l = batch_size * batch + data_index_in_batch
                r = batch_size * batch + data_index_in_batch + 1
                #                 print("before\n",data_index_in_batch)
                feedforward(xtrain[l:r, :], num_hidden)
                #                 print("after\n",y_hat)
                # print(l , r)
                backpropagation(y_true[:, l:r])

                for j in range(len_w):
                    dw[j] += grad_w_k_list[j].transpose()

                for j in range(len_b):
                    db[j] += grad_b[j]

                loss += cal_loss(y_hat, y_true[:, l:r])

                error += cal_error(y_hat, y_true[:, l:r])

                nos_points += 1

            for j in range(len_w):
                weight_matrix_list[j] = weight_matrix_list[j] - eta * dw[j]

            for j in range(len_b):
                bias_list[j] = bias_list[j] - eta * db[j]

            loss += cal_loss(y_hat, y_true[:, l:r])

            if (steps % 300 == 0):
                if(differ_train_val == 1):
                    file.write("Epoch " + str(t + 1) + ", Step " + str(steps) + ", loss : " + str(loss) + ", error : " + str(error / nos_points) + ", lr: " + str(eta) + "\n")
                elif(differ_train_val==0):
                    file1.write("Epoch " + str(t + 1) + ", Step " + str(steps) + ", loss : " + str(loss) + ", error : " + str(error / nos_points) + ", lr: " + str(eta) + "\n")
            steps += 1

        print("end  epochs :  ", t)


    if (differ_train_val==1):
        file.close()
    elif (differ_train_val==0):
        file1.close()


def adam(weight_matrix_list, bias_list, xtrain, num_hidden, sizes, eta, batch_size,epochs,expt_dir, differ_train_val, y_true):
    steps = 0
    loss = 0
    error = 0
    nos_points = 0
    len_w = len(weight_matrix_list)
    len_b = len(bias_list)

    if (differ_train_val == 1):
        file = open(str(expt_dir) + "log_train.txt", 'w')
        print(expt_dir)
    else :
        file = open(str(expt_dir) + "log_val.txt", 'w')

    num_points_seen = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    data_length = len(xtrain)
    no_of_batch = int(data_length / batch_size)

    m_w = []
    m_b = []
    for i in range(int(num_hidden) + 1):
        m_w.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
        m_b.append(0 * np.random.rand(sizes[i + 1], 1))

    v_w = []
    v_b = []
    for i in range(int(num_hidden) + 1):
        v_w.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
        v_b.append(0 * np.random.rand(sizes[i + 1], 1))

    for t in range(epochs):
        save_weights((weight_matrix_list), (epochs))
        for batch in range(no_of_batch):
            dw = []
            db = []
            for i in range(int(num_hidden) + 1):
                dw.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
                db.append(0 * np.random.rand(sizes[i + 1], 1))

            for i in range(batch_size):
                l = batch_size * batch + i
                r = batch_size * batch + i + 1
                feedforward(xtrain[l:r, :], num_hidden)
                # print(l , r)
                backpropagation(y_true[:, l:r])

                for j in range(len_w):
                    dw[j] += grad_w_k_list[j].transpose()

                for j in range(len_b):
                    db[j] += grad_b[j]

                loss += cal_loss(y_hat, y_true[:, l:r])

                error += cal_error(y_hat, y_true[:, l:r])

                nos_points += 1

            for j in range(len_w):
                m_w[j] = beta1 * m_w[j] + (1 - beta1) * dw[j]

            for j in range(len_b):
                m_b[j] = beta1 * m_b[j] + (1 - beta1) * db[j]

            for j in range(len_w):
                v_w[j] = beta2 * v_w[j] + (1 - beta2) * (dw[j] ** 2)

            for j in range(len_b):
                v_b[j] = beta2 * v_b[j] + (1 - beta2) * (db[j] ** 2)

            for j in range(len_w):
                m_w[j] = m_w[j] / (1 - math.pow(beta1, batch + 1))

            for j in range(len_b):
                m_b[j] = m_b[j] / (1 - math.pow(beta1, batch + 1))

            for j in range(len_w):
                v_w[j] = v_w[j] / (1 - math.pow(beta2, batch + 1))

            for j in range(len_b):
                v_b[j] = v_b[j] / (1 - math.pow(beta2, batch + 1))

            for j in range(len_w):
                weight_matrix_list[j] -= (eta / np.sqrt(v_w[j] + eps)) * m_w[j]

            for j in range(len_b):
                bias_list[j] -= (eta / np.sqrt(v_b[j] + eps)) * m_b[j]

            loss += cal_loss(y_hat, y_true[:, l:r])

            if (steps % 300 == 0):
                file.write(
                    "Epoch " + str(t + 1) + ", Step " + str(steps) + ", loss : " + str(loss) + ", error : " + str(error / nos_points) + ", lr: " + str(eta) + "\n")

            steps += 1

        print("end of epoch", t)
        # save_weights((weight_matrix_list), (epochs))
    file.close()


def nestrov_accelerated_gradient_descent(weight_matrix_list, bias_list, xtrain, num_hidden, sizes,eta,batch_size,epochs, expt_dir, differ_train_val, y_true,gamma):
    steps = 0
    loss = 0
    error = 0
    nos_points = 0

    if (differ_train_val == 1):
        file = open(str(expt_dir) + "log_train.txt", 'w')
    else:
        file = open(str(expt_dir) + "log_val.txt", 'w')

    len_w = len(weight_matrix_list)
    len_b = len(bias_list)

    num_points_seen = 0

    t = 0
    data_length = len(xtrain)
    no_of_batch = int(data_length / batch_size)

    prev_v_w = []
    prev_v_b = []
    for i in range(int(num_hidden) + 1):
        prev_v_w.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
        prev_v_b.append(0 * np.random.rand(sizes[i + 1], 1))

    v_w = []
    v_b = []
    for i in range(int(num_hidden) + 1):
        v_w.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
        v_b.append(0 * np.random.rand(sizes[i + 1], 1))

    for t in range(epochs):
        save_weights((weight_matrix_list), (epochs))
        for batch in range(no_of_batch):
            dw = []
            db = []
            for i in range(int(num_hidden) + 1):
                dw.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
                db.append(0 * np.random.rand(sizes[i + 1], 1))

            for j in range(len_w):
                v_w[j] = gamma * prev_v_w[j]

            for j in range(len_b):
                v_b[j] = gamma * prev_v_b[j]

            for j in range(len_w):
                weight_matrix_list[j] -= v_w[j]

            for j in range(len_b):
                bias_list[j] -= v_b[j]

            for i in range(batch_size):
                l = batch_size * batch + i
                r = batch_size * batch + i + 1
                feedforward(xtrain[l:r, :], num_hidden)
                # print(l , r)
                backpropagation(y_true[:, l:r])

                for j in range(len_w):
                    dw[j] += grad_w_k_list[j].transpose()

                for j in range(len_b):
                    db[j] += grad_b[j]

                loss += cal_loss(y_hat, y_true[:, l:r])

                error += cal_error(y_hat, y_true[:, l:r])

                nos_points += 1

            for j in range(len_w):
                v_w[j] = gamma * prev_v_w[j] + eta * dw[j]

            for j in range(len_b):
                v_b[j] = gamma * prev_v_b[j] + eta * db[j]

            for j in range(len_w):
                weight_matrix_list[j] += -1 * eta * dw[j]

            for j in range(len_b):
                bias_list[j] = -1 * eta * db[j]

            for j in range(len_w):
                prev_v_w[j] = v_w[j]

            for j in range(len_b):
                prev_v_b[j] = v_b[j]

            loss += cal_loss(y_hat, y_true[:, l:r])

            if (steps % 300 == 0):
                file.write(
                    "Epoch " + str(t + 1) + ", Step " + str(steps) + ", loss : " + str(loss) + ", error : " + str(
                        error / nos_points) + ", lr: " + str(eta) + "\n")

            steps += 1

        print("end of epoch", t)
        # save_weights((weight_matrix_list), (epochs))
    file.close()


def momentum_gd(weight_matrix_list, bias_list, xtrain, num_hidden, sizes, eta , batch_size, epochs, expt_dir, differ_train_val, y_true,gamma):

    len_w = len(weight_matrix_list)
    len_b = len(bias_list)

    num_points_seen = 0
    t = 0
    data_length = len(xtrain)
    no_of_batch = int(data_length / batch_size)

    steps = 0
    loss = 0
    error = 0
    nos_points = 0

    if (differ_train_val == 1):
        file = open(str(expt_dir) + "log_train.txt", 'w')
    else:
        file = open(str(expt_dir) + "log_val.txt", 'w')


    prev_v_w = []
    prev_v_b = []
    for i in range(int(num_hidden) + 1):
        prev_v_w.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
        prev_v_b.append(0 * np.random.rand(sizes[i + 1], 1))

    v_w = []
    v_b = []
    for i in range(int(num_hidden) + 1):
        v_w.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
        v_b.append(0 * np.random.rand(sizes[i + 1], 1))



    for t in range(epochs):
        save_weights((weight_matrix_list), (epochs))
        for batch in range(no_of_batch):
            dw = []
            db = []
            for i in range(int(num_hidden) + 1):
                dw.append(0 * np.random.rand(sizes[i], sizes[i + 1]))
                db.append(0 * np.random.rand(sizes[i + 1], 1))

            for i in range(batch_size):
                l = batch_size * batch + i
                r = batch_size * batch + i + 1
                feedforward(xtrain[l:r, :], num_hidden)
                # print(l , r)
                backpropagation(y_true[:, l:r])

                for j in range(len_w):
                    dw[j] += grad_w_k_list[j].transpose()

                for j in range(len_b):
                    db[j] += grad_b[j]

                loss += cal_loss(y_hat, y_true[:, l:r])

                error += cal_error(y_hat, y_true[:, l:r])

                nos_points += 1

            for j in range(len_w):
                v_w[j] = gamma * prev_v_w[j] + eta * dw[j]

            for j in range(len_b):
                v_b[j] = gamma * prev_v_b[j] + eta * db[j]

            for j in range(len_w):
                weight_matrix_list[j] += -1 * v_w[j]

            for j in range(len_b):
                bias_list[j] = -1 * v_b[j]

            for j in range(len_w):
                prev_v_w[j] = v_w[j]

            for j in range(len_b):
                prev_v_b[j] = v_b[j]

            loss += cal_loss(y_hat, y_true[:, l:r])

            if (steps % 300 == 0):
                file.write(
                    "Epoch " + str(t + 1) + ", Step " + str(steps) + ", loss : " + str(loss) + ", error : " + str(
                        error / nos_points) + ", lr: " + str(eta) + "\n")

            steps += 1

        print("end of epoch", t)
        # save_weights((weight_matrix_list), (epochs))
    file.close()

if __name__ == '__main__':

    train_data = pd.read_csv(args.train).as_matrix()
    xtrain = train_data[:, 1:785]
    xmean = np.mean(xtrain, axis=1)
    means_expanded = np.outer(xmean, np.ones(784))
    xtrain = xtrain - means_expanded
    xstd = np.std(xtrain, axis=1)
    std_expanded = np.outer(xstd, np.ones(784))
    xtrain = xtrain * 1.0 / std_expanded
    true_label = train_data[:, 785:]
    y_true = np.array(encode_into_prob(true_label, 10))
    y_true = y_true.transpose()

    ip_neurons = 784
    num_hidden = args.num_hidden
    encoding_bits = 10
    sizes = args.sizes
    dummy=[]
    length_size = len(sizes)
    for i in range(length_size):
        if(sizes[i] != ','):
            if(sizes[i] != '0'):
                dummy.append(int(sizes[i]))

    sizes=dummy
    sizes.insert(0, ip_neurons)
    sizes.append(encoding_bits)

    weight_matrix_list = []
    bias_list = []
    y_hat = []

    for i in range(int(num_hidden) + 1):
        weight_matrix_list.append(0.01 * np.random.rand(sizes[i], sizes[i + 1]))
        bias_list.append(0 * np.random.rand(sizes[i + 1], 1))

    eta = args.lr
    batch_size = args.batch_size
    option = args.opt
    max_iteration = args.epochs
    expt_dir = args.expt_dir
    momentum = args.momentum
    save_dir = args.save_dir
    pretratin = args.pretrain
    testing=args.testing
    state=args.state


    print(args.expt_dir)

    # if(testing == False ):
    if (option == "gd"):
        grad_descent(weight_matrix_list, bias_list, xtrain, num_hidden, sizes, eta, batch_size, max_iteration,expt_dir,1,y_true)
    elif (option == "adam"):
        adam(weight_matrix_list,bias_list,xtrain,num_hidden,sizes, eta, batch_size, max_iteration,expt_dir,1,y_true)
    elif (option == "nag"):
        nestrov_accelerated_gradient_descent(weight_matrix_list, bias_list, xtrain, num_hidden, sizes, eta, batch_size,max_iteration,expt_dir,1,y_true,momentum)
    elif (option == "momentum"):
        momentum_gd(weight_matrix_list, bias_list, xtrain, num_hidden, sizes, eta, batch_size, max_iteration,expt_dir,1,y_true,momentum)


    weight_matrix_list = []
    bias_list = []
    y_hat = []

    val_data = pd.read_csv(args.val).as_matrix()
    xval = val_data[:, 1:785]
    xmean = np.mean(xval, axis=1)
    means_expanded = np.outer(xmean, np.ones(784))
    xval = xval - means_expanded
    xstd = np.std(xval, axis=1)
    std_expanded = np.outer(xstd, np.ones(784))
    xval = xval * 1.0 / std_expanded
    true_label_val = val_data[:, 785:]
    y_true_val = np.array(encode_into_prob(true_label_val, 10))
    y_true_val = y_true_val.transpose()

    for i in range(int(num_hidden) + 1):
        weight_matrix_list.append(0.01 * np.random.rand(sizes[i], sizes[i + 1]))
        bias_list.append(0 * np.random.rand(sizes[i + 1], 1))



    if (option == "gd"):
        grad_descent(weight_matrix_list, bias_list, xval, num_hidden, sizes, eta, batch_size, max_iteration, expt_dir,0, y_true_val)


    elif (option == "adam"):

        adam(weight_matrix_list, bias_list, xval , num_hidden, sizes, eta, batch_size, max_iteration, expt_dir, 0, y_true_val)

    elif (option == "nag"):

        nestrov_accelerated_gradient_descent(weight_matrix_list, bias_list, xval, num_hidden, sizes, eta, batch_size, max_iteration, expt_dir, 0, y_true_val, momentum)

    elif (option == "momentum"):

        momentum_gd(weight_matrix_list, bias_list, xval, num_hidden, sizes, eta, batch_size, max_iteration, expt_dir, 0, y_true_val, momentum)

    test_data = pd.read_csv(args.test).as_matrix()
    xval = test_data[:, 1:]
    xmean = np.mean(xval, axis=1)
    means_expanded = np.outer(xmean, np.ones(784))
    xval = xval - means_expanded
    xstd = np.std(xval, axis=1)
    std_expanded = np.outer(xstd, np.ones(784))
    xval = xval * 1.0 / std_expanded

    feedforward(xval, num_hidden)
    y_hat = decode_yhat_to_classes(y_hat)
    y_predict = classifcation(y_hat)
    y_predict = y_predict[:, np.newaxis]
    y_predict = add_id_to_y_predict(y_predict)
    with open(str(expt_dir) + "test submission.csv", 'w', newline='') as f:
        thewriter = csv.writer(f)
        row, col = y_predict.shape
        thewriter.writerow(['id', 'label'])
        for i in range(row):
            thewriter.writerow(y_predict[i])

    # with open(str(expt_dir) + "weights2.pkl", 'w', newline='') as f:
    #     thewriter = csv.writer(f)
    #     #row, col = y_predict.shape
    #     thewriter.writerow(weight_matrix_list)
    #     # rowlen = len(weight_matrix_list)
    #     # for i in range(rowlen):
    #     #     thewriter.writerow(weight_matrix_list[i])