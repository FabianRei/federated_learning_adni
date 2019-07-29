from torch.autograd import Variable
import numpy as np


def batch_gen(data, labels, batchSize, shuffle=True):
    epochData = data.clone()
    epochLabels = labels.clone()
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


def test(batchSize, testData, testLabels, Net, dimIn, includePredictionLabels=False, test_eval=False):
    allAccuracy =[]
    allWrongs = []
    predictions = []
    labels = []
    if test_eval:
        Net.eval()
    for batch_idx, (data, target) in enumerate(batch_gen(testData, testLabels, batchSize, shuffle=False)):
        data_temp = np.copy(data)
        data, target = Variable(data), Variable(target)
        data, target = data.cuda(), target.cuda()
        # data = data.view(-1, dimIn)
        if len(data.shape) == 4:
            data = data.permute(0, 3, 1, 2)
        else:
            data = data.view(-1, 1, dimIn, dimIn)
        # Net.eval()
        net_out = Net(data)
        prediction = net_out.max(1)[1]
        selector = (prediction != target).cpu().numpy().astype(np.bool)
        wrongs = data_temp[selector]
        testAcc = list((prediction == target).cpu().numpy())
        if not sum(testAcc) == len(target) and False:
            print(prediction.cpu().numpy()[testAcc == 0])
            print(target.cpu().numpy()[testAcc==0])
        allAccuracy.extend(testAcc)
        allWrongs.extend(wrongs)
        predictions.extend(prediction)
        labels.extend(target)
    if test_eval:
        Net.train()
    print(f"Test accuracy is {np.mean(allAccuracy)}")
    if includePredictionLabels:
        return np.mean(allAccuracy), np.stack((predictions, testLabels)).T
    else:
        return np.mean(allAccuracy)


def train(epochs, batchSize, trainData, trainLabels, testData, testLabels, Net, test_interval, optimizer, criterion, dimIn):
    bestTestAcc = 0
    testAcc = 0
    Net.train()
    for epoch in range(epochs):
        epochAcc = []
        lossArr = []
        logCount = 0
        testAcc = 0
        for batch_idx, (data, target) in enumerate(batch_gen(trainData, trainLabels, batchSize, shuffle=True)):
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            # data = data.view(-1, dimIn)
            data = data.view(-1, 1, dimIn, dimIn)
            optimizer.zero_grad()
            # Net.train()
            net_out = Net(data)
            prediction = net_out.max(1)[1]
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            currAcc = (prediction == target).cpu().numpy()
            epochAcc.extend(list(currAcc))
            lossArr.append(loss.data.item())
            if logCount % 10 == 0:
                print(f"Train epoch: {epoch} and batch number {logCount}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
            logCount += 1
        print(f"Train epoch: {epoch}, loss is {np.mean(lossArr)}, accuracy is {np.mean(epochAcc)}")
        if epoch % test_interval == 0:
            testAcc = test(batchSize, testData, testLabels, Net, dimIn)
            if testAcc > bestTestAcc:
                bestTestAcc = testAcc
    return Net, testAcc