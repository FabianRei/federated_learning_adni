from torch.autograd import Variable
import numpy as np


def batch_gen(data, labels, batchSize, shuffle=True):
    epochData = data  # .clone()
    epochLabels = labels  # .clone()
    if shuffle:
        selector = np.random.permutation(len(epochData))
        epochData = epochData[selector]
        epochLabels = epochLabels[selector]
    sz = epochData.shape[0]
    i, j = 0, batchSize
    while j < sz + batchSize:
        batchData = epochData[i:min(j, sz)]
        batchLabels = epochLabels[i:min(j, sz)]
        yield batchData, batchLabels
        i += batchSize
        j += batchSize


def test(batchSize, testData, test_labels, Net, dimIn, includePredictionLabels=False, test_eval=True):
    allAccuracy =[]
    allWrongs = []
    predictions = []
    labels = []
    if test_eval:
        Net.eval()
    for batch_idx, (data, target) in enumerate(batch_gen(testData, test_labels, batchSize, shuffle=False)):
        data_temp = np.copy(data)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        # data = data.view(-1, dimIn)
        if len(data.shape) == 3:
            data = data.view(-1, 1, dimIn, dimIn)
        # Net.eval()
        net_out = Net(data)
        prediction = net_out.max(1)[1]
        testAcc = list((prediction == target).cpu().numpy())
        if not sum(testAcc) == len(target) and False:
            print(prediction.cpu().numpy()[testAcc == 0])
            print(target.cpu().numpy()[testAcc==0])
        allAccuracy.extend(testAcc)
        predictions.extend(prediction)
        labels.extend(target)
    if test_eval:
        Net.train()
    print(f"Test accuracy is {np.mean(allAccuracy)}")
    if includePredictionLabels:
        predictions = [int(p) for p in predictions]
        test_labels = [int(l) for l in test_labels]
        return np.mean(allAccuracy), np.stack((predictions, test_labels)).T
    else:
        return np.mean(allAccuracy)


def train(batch_size, train_data, train_labels, test_data, test_labels, Net, optimizer, criterion, test_interval=1,
          epochs=1, dim_in='default'):
    test_acc = 0
    Net.train()
    if dim_in == 'default':
        dim_in = train_data.shape[-1]
    for epoch in range(epochs):
        epoch_acc = []
        loss_arr = []
        logCount = 0
        test_acc = -1
        train_predictions = []
        train_target = []
        for batch_idx, (data, target) in enumerate(batch_gen(train_data, train_labels, batch_size, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            # data = data.view(-1, dimIn)
            if len(data.shape) == 3:
                data = data.view(-1, 1, dim_in, dim_in)
            optimizer.zero_grad()
            # Net.train()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            batch_acc = (prediction == target).cpu().numpy()
            train_predictions.extend(prediction)
            train_target.extend(target)
            epoch_acc.extend(list(batch_acc))
            loss_arr.append(loss.data.item())

            if logCount % 10 == 0:
                print(f"Train epoch: {epoch} and batch number {logCount}, loss is {np.mean(loss_arr)}, accuracy is {np.mean(epoch_acc)}")
            logCount += 1
        print(f"Train epoch: {epoch}, loss is {np.mean(loss_arr)}, accuracy is {np.mean(epoch_acc)}")
        if epoch % test_interval == 0 and test_interval > 0:
            optimizer.zero_grad()
            test_acc, test_pred_label = test(batch_size, test_data, test_labels, Net, dim_in, includePredictionLabels=True)
        train_target = [int(t) for t in train_target]
        train_predictions = [int(p) for p in train_predictions]
    return Net, test_acc, test_pred_label, np.mean(epoch_acc), np.mean(loss_arr), np.stack((train_target, train_predictions)).T
