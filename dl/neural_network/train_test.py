from torch.autograd import Variable
import numpy as np
import fnmatch


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
    softmax_predictions = []
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
        softmax_prediction = np.exp(net_out.cpu().detach().numpy())
        testAcc = list((prediction == target).cpu().numpy())
        if not sum(testAcc) == len(target) and False:
            print(prediction.cpu().numpy()[testAcc == 0])
            print(target.cpu().numpy()[testAcc==0])
        allAccuracy.extend(testAcc)
        predictions.extend(prediction)
        softmax_predictions.extend(list(softmax_prediction))
        labels.extend(target)
    if test_eval:
        Net.train()
    print(f"Test accuracy is {np.mean(allAccuracy)}")
    if includePredictionLabels:
        predictions = np.array([int(p) for p in predictions])
        predictions = np.expand_dims(predictions, axis=1)
        test_labels = np.array([int(l) for l in test_labels])
        test_labels = np.expand_dims(test_labels, axis=1)
        softmax_predictions = np.array(softmax_predictions)
        return np.mean(allAccuracy), np.concatenate((predictions, test_labels, softmax_predictions), axis=1)
    else:
        return np.mean(allAccuracy)


def train(batch_size, train_data, train_labels, test_data, test_labels, Net, optimizer, criterion, test_interval=1,
          epochs=1, dim_in='default'):
    test_acc = 0
    is_resnext = fnmatch.fnmatch(type(Net).__name__, '*ResNext*')
    is_resnet152 = fnmatch.fnmatch(type(Net).__name__, '*ResNet152*')
    is_resnet152 = False
    aggregation_number = int(batch_size // 8)
    aggregation_count = 0
    if is_resnext:
        # max possible batch size is 4. We aggregate and pass back for set batch_size.
        batch_size = 8
    if is_resnet152:
        batch_size = 32
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
        optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(batch_gen(train_data, train_labels, batch_size, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            # import pdb; pdb.set_trace()
            # data = data.view(-1, dimIn)
            if len(data.shape) == 3:
                data = data.view(-1, 1, dim_in, dim_in)
            # Net.train()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            loss.backward()
            if is_resnext or is_resnet152:
                if aggregation_count >= batch_size*aggregation_number:
                    optimizer.step()
                    optimizer.zero_grad()
                    aggregation_count = 0
                else:
                    aggregation_count += batch_size
            else:
                optimizer.step()
                optimizer.zero_grad()
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
