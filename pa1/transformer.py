import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    W_Q, W_K, W_V, W_O = nodes[0], nodes[1], nodes[2], nodes[3]
    W1, b1, W2, b2 = nodes[4], nodes[5], nodes[6], nodes[7]
    # W_proj, b_proj = nodes[8], nodes[9]
    
    Q = ad.matmul(X, W_Q)
    K = ad.matmul(X, W_K)
    V = ad.matmul(X, W_V)

    # Attention
    attn_scores = ad.matmul(Q, ad.transpose(K, -1, -2)) / (model_dim ** 0.5)
    attn_probs = ad.softmax(attn_scores, dim=-1)
    self_attn = ad.matmul(attn_probs, V)
    self_attn = ad.matmul(self_attn, W_O)
    
    # Feed Forward
    ff = ad.matmul(self_attn, W1) + b1
    ff = ad.relu(ff)
    ff = ad.matmul(ff, W2) + b2
    
    output = ad.mean(ff, dim=1)
    return output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    softmax = ad.softmax(Z)
    
    # Cross Entropy
    loss = ad.mul_by_const(ad.sum_op(y_one_hot * ad.log(softmax), dim=1), -1)
    loss = ad.mean(loss, dim=0, keepdim=True)
    return loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    param_names = [
        'W_Q', 'W_K', 'W_V', 'W_O',
        'W1', 'b1', 'W2', 'b2',
        'W_linear', 'b_linear'
    ]
    
    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        logits, loss_val, *grads = f_run_model({
            'X': X_batch,
            'y': y_batch,
            **dict(zip(param_names, model_weights))
        })

        # Update weights and biases
        # Hint: You can update the tensor using something like below:
        # W_Q -= lr * grad_W_Q.sum(dim=0)
        with torch.no_grad():
            for j, param in enumerate(model_weights):
                if 'b' in param_names[j]:  # Handle bias terms differently
                    param -= lr * grads[j].mean(dim=(0, 1))
                else:
                    param -= lr * grads[j].mean(dim=0)

        # Accumulate the loss
        total_loss += loss_val.item() * (end_idx - start_idx)

    # Compute the average loss
    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)
    return model_weights, average_loss



def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 10
    batch_size = 50
    lr = 0.01

    # Set up model params
    weights_name_map = {
        'W_Q': ad.Variable('W_Q'),
        'W_K': ad.Variable('W_K'),
        'W_V': ad.Variable('W_V'),
        'W_O': ad.Variable('W_O'),
        'W1': ad.Variable('W1'),
        'b1': ad.Variable('b1'),
        'W2': ad.Variable('W2'),
        'b2': ad.Variable('b2'),
        'W_linear': ad.Variable('W_linear'),
        'b_linear': ad.Variable('b_linear'),
    }
    transformer_weights_variables = [
        v for k, v in weights_name_map.items() 
        if k != 'W_linear' and k != 'b_linear'
    ]

    # TODO: Define the forward graph.
    X = ad.Variable('X')
    X_transformer = transformer(X, transformer_weights_variables, 
                        model_dim, seq_length, eps, batch_size, num_classes)
    y_predict: ad.Node = ad.matmul(X_transformer, weights_name_map['W_linear']) + weights_name_map['b_linear']
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # TODO: Construct the backward graph.
    grads: List[ad.Node] = ad.gradients(loss, list(weights_name_map.values()))
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))
    W_linear_val = np.random.uniform(-stdv, stdv, (num_classes, num_classes))
    b_linear_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        # Convert str:tensor to Variable:tensor
        input_values = {
            X: model_weights['X'],
            y_groundtruth: model_weights['y'],
            **{weights_name_map[name]: value 
               for name, value in model_weights.items() 
               if name != 'X' and name != 'y'}
        }
        
        result = evaluator.run(
            input_values=input_values
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run({
                X: X_batch,
                **dict(zip(weights_name_map.values(), model_weights))
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        all_logits = [tensor.detach().numpy() for tensor in all_logits]
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val, requires_grad=True),
        torch.tensor(W_K_val, requires_grad=True),
        torch.tensor(W_V_val, requires_grad=True),
        torch.tensor(W_O_val, requires_grad=True),
        torch.tensor(W_1_val, requires_grad=True),
        torch.tensor(b_1_val, requires_grad=True),
        torch.tensor(W_2_val, requires_grad=True),
        torch.tensor(b_2_val, requires_grad=True),
        torch.tensor(W_linear_val, requires_grad=True),
        torch.tensor(b_linear_val, requires_grad=True),
    ]
    
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
