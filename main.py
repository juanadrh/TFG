import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import randn



def sigma(z, forward=True):
    if forward:
        return 1 / (1 + np.exp(-z))
    else:
        return z * (1 - z)


fig, ax = plt.subplots(figsize=(10, 3))

# Solid line.
x = np.linspace(-5, 5)
y = sigma(x)
ax.plot(x, y, color='C9', lw=3)

# Dotted line.
x_ = np.linspace(-6, 6, 80)
y_ = sigma(x_)
ax.plot(x_, y_, '--', color='C9',lw=3)

ax.axvline(0, color='k', lw=0.75, zorder=0)
ax.grid(color='k', alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=14)

# plt.show()


def forward(xi, W1, b1, W2, b2):
    z1 = W1 @ xi + b1
    a1 = sigma(z1)
    z2 = W2 @ a1 + b2
    return z2, a1


def backward(xi, yi,
             a1, z2,
             params,
             learning_rate):
    err_output = z2 - yi
    grad_W2 = err_output * a1
    params['W2'] -= learning_rate * grad_W2

    grad_b2 = err_output
    params['b2'] -= learning_rate * grad_b2

    derivative = sigma(a1, forward=False)
    err_hidden = err_output * derivative * params['W2']
    grad_W1 = err_hidden[:, None] @ xi[None, :]
    params['W1'] -= learning_rate * grad_W1

    grad_b1 = err_hidden
    params['b1'] -= learning_rate * grad_b1

    return params


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')


def initialize_params(units, features):
    np.random.seed(42)
    params = {
        "W1": 0.1 * randn(units, features),
        "b1": np.zeros(shape=units),

        "W2": 0.1 * randn(units),
        "b2": np.zeros(shape=1)
    }
    return params


units = 300
features = X_train.shape[-1]

params = initialize_params(units, features)

pre_train = []
for xi in X_train:
    y_pred, _ = forward(xi, **params)
    pre_train.append(y_pred.item())

pre_val = []
for xi in X_val:
    y_pred, _ = forward(xi, **params)
    pre_val.append(y_pred.item())


# Be sure to re-run this before re-training the network!
params = initialize_params(units, features)

num_epochs = 100
learning_rate = 0.001
loss_history = []

data = list(zip(X_train, y_train))

num_epochs = 100
learning_rate = 0.001
loss_history = []

data = list(zip(X_train, y_train))

for i in tqdm(range(num_epochs)):

    np.random.shuffle(data)
    loss = 0

    for xi, yi in data:
        z2, a1 = forward(xi, **params)

        params = backward(xi, yi,
                          a1, z2,
                          params,
                          learning_rate)

        loss += np.square(z2 - yi)

    loss_history.append(loss / y_train.size)

params = initialize_params(units, features)

num_epochs = 100
learning_rate = 0.001
loss_history, loss_val_history = [], []

data = list(zip(X_train, y_train))
data_val = list(zip(X_val, y_val))

for i in tqdm(range(num_epochs)):

    # Validation.
    # We do this first to get the same training state
    # as for the training data (below).
    np.random.shuffle(data_val)
    loss = 0

    for xi, yi in data_val:
        z2, a1 = forward(xi, **params)
        loss += np.square(z2 - yi)

    loss_val_history.append(loss / y_val.size)

    # Training.
    np.random.shuffle(data)
    loss = 0

    for xi, yi in data:
        z2, a1 = forward(xi, **params)

        params = backward(xi, yi,
                          a1, z2,
                          params,
                          learning_rate)

        loss += np.square(z2 - yi)

    loss_history.append(loss / y_train.size)


fig, ax = plt.subplots(figsize=(10,3))

ax.semilogy(loss_history, label='Training loss')
ax.semilogy(loss_val_history, label='Validation loss')


#ax.set_ylim(0, 0.0075)
ax.text(-3, 0.02, 'Mean squared error vs epoch number', fontsize=16)
# ax.set_xlabel('Epoch number', fontsize=14)
# ax.set_ylabel('Mean squared error', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid()
ax.legend(fontsize=14)

plt.tight_layout()
plt.show()

post_train = []
for xi in X_train:
    y_pred, _ = forward(xi, **params)
    post_train.append(y_pred.item())


fig, ax = plt.subplots(figsize=(10,3))

ax.plot(y_train[:100], label='truth')
ax.plot(post_train[:100], label='predictions')
ax.plot(pre_train[:100], label='untrained', lw=0.75)

ax.text(-3, 0.19, 'Neural network training data', fontsize=16, ha='left')

mse = loss_history[-1].item()
ax.text(-3, -0.24,
        f'MSE: {mse:.2e}',
        fontsize=14,
        va='center', ha='left')

ax.set_xlabel('Data instance number', fontsize=14)
ax.set_ylabel('y (output signal)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.text(98, 0.195, "Truth", color='C0', fontsize=14, va='top', ha='right')
ax.text(98, 0.14, "Predictions", color='C1', fontsize=14, va='top', ha='right')
ax.text(99, -0.24, "Untrained", color='C2', fontsize=14, va='top', ha='right')

ax.grid(color='k', alpha=0.3)

plt.tight_layout()
plt.show()

post_val = []
for xi in X_val:
    y_pred, _ = forward(xi, **params)
    post_val.append(y_pred.item())


fig, ax = plt.subplots(figsize=(10,3))

ax.plot(y_val, label='truth')
ax.plot(post_val, label='predictions')
#ax.plot(pre_val, label='untrained', lw=0.75)

ax.text(-3, 0.12,
        'Neural network validation',
        fontsize=16,
        va='center', ha='left')

mse = loss_val_history[-1].item()
ax.text(-3, -0.32,
        f'MSE: {mse:.2e}',
        fontsize=14,
        va='center', ha='left')

ax.set_xlabel('Data instance number', fontsize=14)
ax.set_ylabel('y (output signal)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.text(99, 0.1, "Truth", color='C0', fontsize=14, va='top', ha='right')
ax.text(99, -0.08, "Predictions", color='C1', fontsize=14, va='top', ha='right')
#ax.text(99, -0.25, "Untrained", color='C2', fontsize=14, va='top', ha='right')

ax.grid(color='k', alpha=0.3)

plt.tight_layout()
plt.show()


# BLIND TEST: New Rocks
X_blind = np.load('X_blind.npy')
y_blind = np.load('y_blind.npy')

blind_pre = []
for xi in X_blind:
    y_pred, _ = forward(xi, **initialize_params(units, features))
    blind_pre.append(y_pred.item())

blind_train = []
for xi in X_blind:
    y_pred, _ = forward(xi, **params)
    blind_train.append(y_pred.item())

fig, ax = plt.subplots(figsize=(10,3))

ax.plot(y_blind.real[:100], label='Truth')
ax.plot(blind_train[:100], label='Predictions')
#ax.plot(blind_pre[:100], label='Untrained', lw=0.75)

ax.text(-3, 0.075, 'Neural network blind test', fontsize=16, ha='left')

mse = np.mean(np.square(blind_train - y_blind))

ax.text(-3, -0.1,
        f'MSE: {mse:.2e}',
        fontsize=14,
        va='center', ha='left')

ax.set_xlabel('Data instance number', fontsize=14)
ax.set_ylabel('y (output signal)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)

ax.text(99, 0.06, "Truth", color='C0', fontsize=14, va='bottom', ha='right')
ax.text(99, -0.07, "Predictions", color='C1', fontsize=14, va='top', ha='right')
#ax.text(99, -0.23, "Untrained", color='C2', fontsize=14, va='top', ha='right')

ax.grid(color='k', alpha=0.2)

plt.tight_layout()
plt.show()
