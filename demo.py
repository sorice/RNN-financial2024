from rnn import NNTrainer
import matplotlib.pyplot as plt
from logger import logger

data_dir = "dataset/nasdaq100/"

model = NNTrainer(file_data="{}small/nasdaq100_padding.csv".format(data_dir), logger=logger, parallel=False, learning_rate=.001)

model.train(n_epochs=100)

y_pred = model.predict()

plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
plt.show()

plt.figure()
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
plt.show()

plt.figure()
plt.plot(y_pred, label = "Predicted")
plt.plot(model.y[model.train_size:], label = "True")
plt.legend(loc = "upper left")
plt.show()