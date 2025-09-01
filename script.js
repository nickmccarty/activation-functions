const activationFunctions = [
    {
        id: 'relu',
        name: 'ReLU',
        fullName: 'Rectified Linear Unit',
        description: 'The most widely used activation function that outputs the input directly if positive, otherwise outputs zero.',
        formula: 'f(x) = max(0, x)',
        category: 'common',
        tags: ['fast', 'simple', 'gradient-friendly'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.ReLU()
relu = nn.ReLU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = relu(x)
print(output)  # tensor([0., 0., 0., 1., 2.])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.relu(x)

# Method 3: Using inplace operation
relu_inplace = nn.ReLU(inplace=True)`,
        tensorflow_code: `import tensorflow as tf
import numpy as np

# Method 1: Using tf.keras.activations.relu
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.relu(x)
print(output)  # tf.Tensor([0. 0. 0. 1. 2.], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.relu
output = tf.nn.relu(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    # or equivalently:
    # tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
])

# Method 4: As separate layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.ReLU()
])

# Method 5: Custom layer with ReLU
class CustomLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.relu(inputs)`,
        properties: {
            'Output Range': '[0, ∞)',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-check-circle',
                title: 'Good for deep networks',
                description: 'ReLU helps mitigate vanishing gradient problem in deep networks'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Watch for dying ReLU',
                description: 'Neurons can become inactive if they consistently output zero'
            },
            {
                icon: 'fas fa-bolt',
                title: 'Computationally efficient',
                description: 'Simple thresholding operation makes it very fast'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Good gradient flow',
                description: 'Gradient is 1 for positive inputs, enabling good backpropagation'
            }
        ]
    },
    {
        id: 'sigmoid',
        name: 'Sigmoid',
        fullName: 'Sigmoid Function',
        description: 'Maps input values to a range between 0 and 1, creating an S-shaped curve.',
        formula: 'f(x) = 1 / (1 + e^(-x))',
        category: 'common',
        tags: ['probability', 'smooth', 'saturating'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Sigmoid()
sigmoid = nn.Sigmoid()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = sigmoid(x)
print(output)  # tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.sigmoid(x)

# Common use in binary classification
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.activations.sigmoid
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.sigmoid(x)
print(output)  # tf.Tensor([0.1192 0.2689 0.5000 0.7311 0.8808], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.sigmoid
output = tf.nn.sigmoid(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid')
])

# Method 4: Binary classification example
class BinaryClassifier(tf.keras.Model):
    def __init__(self, input_size):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        return self.dense(x)`,
        properties: {
            'Output Range': '[0, 1]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-percentage',
                title: 'Perfect for probabilities',
                description: 'Output range [0,1] makes it ideal for binary classification'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Vanishing gradient problem',
                description: 'Gradients become very small for large positive/negative inputs'
            },
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth and differentiable',
                description: 'Provides smooth gradients everywhere'
            },
            {
                icon: 'fas fa-balance-scale',
                title: 'Not zero-centered',
                description: 'Can cause optimization difficulties in hidden layers'
            }
        ]
    },
    {
        id: 'tanh',
        name: 'Tanh',
        fullName: 'Hyperbolic Tangent',
        description: 'Similar to sigmoid but outputs values between -1 and 1, making it zero-centered.',
        formula: 'f(x) = (e^x - e^(-x)) / (e^x + e^(-x))',
        category: 'common',
        tags: ['zero-centered', 'smooth', 'saturating'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Tanh()
tanh = nn.Tanh()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tanh(x)
print(output)  # tensor([-0.9640, -0.7616, 0.0000, 0.7616, 0.9640])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.tanh(x)

# Common use in RNNs
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden):
        return self.tanh(self.i2h(x) + self.h2h(hidden))`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.activations.tanh
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.tanh(x)
print(output)  # tf.Tensor([-0.964 -0.7616 0. 0.7616 0.964], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.tanh
output = tf.nn.tanh(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh')
])

# Method 4: RNN example with tanh
class SimpleRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.i2h = tf.keras.layers.Dense(hidden_size)
        self.h2h = tf.keras.layers.Dense(hidden_size, use_bias=False)
    
    def call(self, x, hidden):
        return tf.nn.tanh(self.i2h(x) + self.h2h(hidden))`,
        properties: {
            'Output Range': '[-1, 1]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'Yes'
        },
        checklist: [
            {
                icon: 'fas fa-balance-scale',
                title: 'Zero-centered output',
                description: 'Better than sigmoid for hidden layers due to zero-centering'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Still has vanishing gradients',
                description: 'Suffers from vanishing gradients like sigmoid'
            },
            {
                icon: 'fas fa-robot',
                title: 'Good for RNNs',
                description: 'Commonly used in recurrent neural networks'
            },
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth everywhere',
                description: 'Differentiable across entire input range'
            }
        ]
    },
    {
        id: 'leakyrelu',
        name: 'LeakyReLU',
        fullName: 'Leaky Rectified Linear Unit',
        description: 'Modified ReLU that allows small negative values to pass through, preventing dying neurons.',
        formula: 'f(x) = max(αx, x) where α = 0.01',
        category: 'common',
        tags: ['gradient-friendly', 'non-zero-negative'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.LeakyReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = leaky_relu(x)
print(output)  # tensor([-0.0200, -0.0100, 0.0000, 1.0000, 2.0000])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.leaky_relu(x, negative_slope=0.01)

# Custom negative slope
leaky_relu_custom = nn.LeakyReLU(negative_slope=0.1)
output_custom = leaky_relu_custom(x)`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.layers.LeakyReLU
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = leaky_relu(x)
print(output)  # tf.Tensor([-0.02 -0.01 0. 1. 2.], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.leaky_relu
output = tf.nn.leaky_relu(x, alpha=0.01)

# Method 3: As layer in model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.LeakyReLU(alpha=0.01)
])

# Method 4: Custom negative slope
leaky_relu_custom = tf.keras.layers.LeakyReLU(alpha=0.1)
output_custom = leaky_relu_custom(x)`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-heartbeat',
                title: 'Prevents dying ReLU',
                description: 'Small gradient for negative inputs keeps neurons alive'
            },
            {
                icon: 'fas fa-fast-forward',
                title: 'Still computationally fast',
                description: 'Nearly as efficient as standard ReLU'
            },
            {
                icon: 'fas fa-sliders-h',
                title: 'Tunable parameter',
                description: 'Negative slope can be adjusted (typically 0.01-0.1)'
            },
            {
                icon: 'fas fa-arrow-up',
                title: 'Good for deep networks',
                description: 'Better gradient flow than standard ReLU'
            }
        ]
    },
    {
        id: 'gelu',
        name: 'GELU',
        fullName: 'Gaussian Error Linear Unit',
        description: 'Smooth approximation to ReLU that has become popular in transformer architectures.',
        formula: 'f(x) = x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))',
        category: 'advanced',
        tags: ['smooth', 'transformer', 'probabilistic'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.GELU()
gelu = nn.GELU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = gelu(x)
print(output)  # tensor([-0.0454, -0.1587, 0.0000, 0.8413, 1.9545])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.gelu(x)

# Common use in transformers
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.activations.gelu
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.gelu(x)
print(output)  # tf.Tensor([-0.0454 -0.1587 0. 0.8413 1.9545], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.gelu
output = tf.nn.gelu(x)

# Method 3: As activation in Dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='gelu')
])

# Method 4: Transformer block example
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(d_ff, activation='gelu')
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=None):
        return self.linear2(self.dropout(self.linear1(x), training=training))`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-robot',
                title: 'Transformer favorite',
                description: 'Widely used in BERT, GPT, and other transformer models'
            },
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth everywhere',
                description: 'Provides smooth gradients unlike ReLU'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Better than ReLU variants',
                description: 'Often outperforms ReLU and variants in deep networks'
            },
            {
                icon: 'fas fa-calculator',
                title: 'Computationally expensive',
                description: 'More complex calculation than ReLU family'
            }
        ]
    },
    {
        id: 'elu',
        name: 'ELU',
        fullName: 'Exponential Linear Unit',
        description: 'Smooths negative values using exponential function, reducing bias shift.',
        formula: 'f(x) = x if x > 0, α(e^x - 1) if x ≤ 0',
        category: 'advanced',
        tags: ['smooth', 'negative-saturation', 'bias-reducing'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.ELU()
elu = nn.ELU(alpha=1.0)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = elu(x)
print(output)  # tensor([-0.8647, -0.6321, 0.0000, 1.0000, 2.0000])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.elu(x, alpha=1.0)

# Custom alpha parameter
elu_custom = nn.ELU(alpha=1.5)
output_custom = elu_custom(x)`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.layers.ELU
elu = tf.keras.layers.ELU(alpha=1.0)
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = elu(x)
print(output)  # tf.Tensor([-0.8647 -0.6321 0. 1. 2.], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.elu
output = tf.nn.elu(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.ELU(alpha=1.0)
])

# Method 4: Custom alpha parameter
elu_custom = tf.keras.layers.ELU(alpha=1.5)
output_custom = elu_custom(x)`,
        properties: {
            'Output Range': '(-α, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'Nearly'
        },
        checklist: [
            {
                icon: 'fas fa-balance-scale',
                title: 'Reduces bias shift',
                description: 'Negative outputs help center activations around zero'
            },
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth activation',
                description: 'Differentiable everywhere, including at zero'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Computationally expensive',
                description: 'Exponential calculation is slower than ReLU'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Can improve convergence',
                description: 'Often converges faster than ReLU in practice'
            }
        ]
    },
    {
        id: 'selu',
        name: 'SELU',
        fullName: 'Scaled Exponential Linear Unit',
        description: 'Self-normalizing activation that maintains mean and variance through layers.',
        formula: 'f(x) = λ * ELU(x) where λ ≈ 1.0507',
        category: 'advanced',
        tags: ['self-normalizing', 'theoretical', 'deep-networks'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.SELU()
selu = nn.SELU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = selu(x)
print(output)  # tensor([-0.9088, -0.6640, 0.0000, 1.0507, 2.1014])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.selu(x)

# SELU networks require specific initialization
class SELUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SELU(),
            nn.AlphaDropout(0.1),  # Use AlphaDropout with SELU
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.activations.selu
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.selu(x)
print(output)  # tf.Tensor([-0.9088 -0.664 0. 1.0507 2.1014], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.selu
output = tf.nn.selu(x)

# Method 3: As activation function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='selu')
])

# Method 4: SELU network with proper dropout
class SELUNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Use lecun_normal initialization for SELU
        self.layers = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='selu',
                                kernel_initializer='lecun_normal'),
            tf.keras.layers.AlphaDropout(0.1),  # Use AlphaDropout with SELU
            tf.keras.layers.Dense(hidden_size, activation='selu',
                                kernel_initializer='lecun_normal'),
            tf.keras.layers.Dense(num_classes)
        ])
    
    def call(self, x):
        return self.layers(x)`,
        properties: {
            'Output Range': '(-λα, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'Self-normalizing'
        },
        checklist: [
            {
                icon: 'fas fa-balance-scale',
                title: 'Self-normalizing',
                description: 'Maintains mean=0, variance=1 through network layers'
            },
            {
                icon: 'fas fa-layer-group',
                title: 'No batch normalization needed',
                description: 'Self-normalization eliminates need for BatchNorm'
            },
            {
                icon: 'fas fa-cog',
                title: 'Requires specific setup',
                description: 'Needs proper weight initialization and AlphaDropout'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Good for very deep networks',
                description: 'Theoretical guarantees for deep architectures'
            }
        ]
    },
    {
        id: 'glu',
        name: 'GLU',
        fullName: 'Gated Linear Unit',
        description: 'Uses gating mechanism to control information flow, popular in language models.',
        formula: 'f(x) = (X * W + b) ⊗ σ(X * V + c)',
        category: 'advanced',
        tags: ['gating', 'language-models', 'attention'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.GLU()
glu = nn.GLU(dim=-1)
# Input must have even number of features (splits in half)
x = torch.randn(2, 8)  # Batch size 2, 8 features
output = glu(x)  # Output: (2, 4) - half the input size
print(output.shape)

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.glu(x, dim=-1)

# Common use in language models
class GLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear = nn.Linear(d_model, d_ff * 2)  # 2x for gating
        self.glu = nn.GLU(dim=-1)
    
    def forward(self, x):
        return self.glu(self.linear(x))`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'No',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-door-open',
                title: 'Gating mechanism',
                description: 'Controls information flow through learned gates'
            },
            {
                icon: 'fas fa-comment',
                title: 'Great for NLP',
                description: 'Widely used in language models and transformers'
            },
            {
                icon: 'fas fa-expand-arrows-alt',
                title: 'Doubles parameter count',
                description: 'Requires 2x parameters compared to simple activations'
            },
            {
                icon: 'fas fa-brain',
                title: 'Learns what to activate',
                description: 'More expressive than fixed activation functions'
            }
        ]
    },
    {
        id: 'prelu',
        name: 'PReLU',
        fullName: 'Parametric ReLU',
        description: 'Learnable version of LeakyReLU where the negative slope is a trainable parameter.',
        formula: 'f(x) = max(αx, x) where α is learnable',
        category: 'advanced',
        tags: ['learnable', 'gradient-friendly', 'adaptive'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.PReLU()
prelu = nn.PReLU(num_parameters=1)  # Single parameter for all channels
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = prelu(x)
print(f"Output: {output}")
print(f"Learned parameter: {prelu.weight.item()}")

# Method 2: Channel-wise parameters
prelu_channelwise = nn.PReLU(num_parameters=64)  # For 64 channels

# Example network
class PReLUNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.prelu1 = nn.PReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.prelu2 = nn.PReLU()
    
    def forward(self, x):
        x = self.prelu1(self.linear1(x))
        x = self.prelu2(self.linear2(x))
        return x`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.layers.PReLU()
prelu = tf.keras.layers.PReLU()
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
# Note: PReLU needs to be called within a model
model = tf.keras.Sequential([prelu])
output = model(x)
print(f"Output: {output}")

# Method 2: Channel-wise PReLU
prelu_channelwise = tf.keras.layers.PReLU(
    shared_axes=[1, 2]  # Share across height and width for conv layers
)

# Example network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(32),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(1)
])`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-graduation-cap',
                title: 'Learns optimal slope',
                description: 'Automatically finds best negative slope during training'
            },
            {
                icon: 'fas fa-heartbeat',
                title: 'Prevents dying neurons',
                description: 'Like LeakyReLU but with adaptive slope'
            },
            {
                icon: 'fas fa-plus',
                title: 'Extra parameters',
                description: 'Adds learnable parameters to your model'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Good empirical results',
                description: 'Often outperforms fixed LeakyReLU'
            }
        ]
    },
    {
        id: 'swish',
        name: 'SiLU/Swish',
        fullName: 'Sigmoid Linear Unit / Swish',
        description: 'Smooth activation that multiplies input by its sigmoid, popular in modern architectures.',
        formula: 'f(x) = x * σ(x) = x / (1 + e^(-x))',
        category: 'advanced',
        tags: ['smooth', 'modern', 'self-gating'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.SiLU() (PyTorch's implementation)
silu = nn.SiLU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = silu(x)
print(output)  # tensor([-0.2384, -0.2689, 0.0000, 0.7311, 1.7616])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.silu(x)

# Manual implementation (equivalent to SiLU/Swish)
def swish(x):
    return x * torch.sigmoid(x)

# Common use in modern architectures
class ModernBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.silu = nn.SiLU()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(self.silu(self.linear(x)))`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.activations.swish
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.swish(x)
print(output)  # tf.Tensor([-0.2384 -0.2689 0. 0.7311 1.7616], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.swish
output = tf.nn.swish(x)

# Method 3: Manual implementation
def swish_custom(x):
    return x * tf.keras.activations.sigmoid(x)

# Method 4: As activation function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='swish')
])

# Method 5: In modern architectures
class ModernBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(dim, activation='swish')
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        return self.norm(self.dense(x))`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth everywhere',
                description: 'Differentiable across entire input range'
            },
            {
                icon: 'fas fa-door-open',
                title: 'Self-gating property',
                description: 'Acts as its own gate through sigmoid multiplication'
            },
            {
                icon: 'fas fa-star',
                title: 'Modern architecture favorite',
                description: 'Used in EfficientNet, MobileNet, and many recent models'
            },
            {
                icon: 'fas fa-calculator',
                title: 'More expensive than ReLU',
                description: 'Requires sigmoid computation'
            }
        ]
    },
    {
        id: 'mish',
        name: 'Mish',
        fullName: 'Mish Activation',
        description: 'Self-regularized activation function that often outperforms ReLU and Swish.',
        formula: 'f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))',
        category: 'advanced',
        tags: ['smooth', 'self-regularizing', 'unbounded'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Mish()
mish = nn.Mish()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = mish(x)
print(output)  # tensor([-0.2525, -0.3034, 0.0000, 0.8651, 1.9440])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.mish(x)

# Manual implementation
def mish_manual(x):
    return x * torch.tanh(F.softplus(x))

# Example usage in CNN
class MishCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.mish1 = nn.Mish()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.mish2 = nn.Mish()
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.mish1(self.conv1(x))
        x = self.mish2(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'Nearly'
        },
        checklist: [
            {
                icon: 'fas fa-shield-alt',
                title: 'Self-regularizing',
                description: 'Built-in regularization properties reduce overfitting'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Strong empirical results',
                description: 'Often outperforms ReLU, Swish in computer vision'
            },
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth negative region',
                description: 'Better gradient flow than ReLU variants'
            },
            {
                icon: 'fas fa-calculator',
                title: 'Computationally intensive',
                description: 'Most expensive among common activations'
            }
        ]
    },
    {
        id: 'softmax',
        name: 'Softmax',
        fullName: 'Softmax Function',
        description: 'Converts a vector of values into a probability distribution that sums to 1.',
        formula: 'f(x_i) = e^(x_i) / Σ(e^(x_j)) for j=1 to K',
        category: 'common',
        tags: ['probability', 'multiclass', 'normalization'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Softmax()
softmax = nn.Softmax(dim=-1)
x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
output = softmax(x)
print(output)  # Each row sums to 1.0

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.softmax(x, dim=-1)

# Common use in classification
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        # Note: No softmax here if using CrossEntropyLoss
    
    def forward(self, x):
        logits = self.linear(x)
        # Apply softmax only for inference, not during training
        if not self.training:
            return F.softmax(logits, dim=-1)
        return logits`,
        tensorflow_code: `import tensorflow as tf

# Method 1: Using tf.keras.activations.softmax
x = tf.constant([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
output = tf.keras.activations.softmax(x)
print(output)  # Each row sums to 1.0

# Method 2: Using tf.nn.softmax
output = tf.nn.softmax(x, axis=-1)

# Method 3: As activation in Dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='softmax')
])

# Method 4: Classification example
class Classifier(tf.keras.Model):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Don't use softmax if using sparse_categorical_crossentropy
        self.dense = tf.keras.layers.Dense(num_classes)
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, x, training=None):
        logits = self.dense(x)
        # Apply softmax only for inference
        if not training:
            return self.softmax(logits)
        return logits`,
        properties: {
            'Output Range': '[0, 1]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes (order)',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-percentage',
                title: 'Perfect for probabilities',
                description: 'Outputs sum to 1, ideal for multiclass classification'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Only for output layers',
                description: 'Rarely used in hidden layers due to saturation'
            },
            {
                icon: 'fas fa-fire',
                title: 'Temperature scaling',
                description: 'Can adjust sharpness with temperature parameter'
            },
            {
                icon: 'fas fa-calculator',
                title: 'Expensive computation',
                description: 'Requires exponentials and normalization'
            }
        ]
    },
    {
        id: 'hardswish',
        name: 'Hardswish',
        fullName: 'Hard Swish',
        description: 'Computationally efficient approximation of Swish, designed for mobile devices.',
        formula: 'f(x) = x * ReLU6(x + 3) / 6',
        category: 'advanced',
        tags: ['efficient', 'mobile', 'approximation'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Hardswish()
hardswish = nn.Hardswish()
x = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
output = hardswish(x)
print(output)  # tensor([0.0000, -0.3333, 0.0000, 1.6667, 4.0000])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.hardswish(x)

# Manual implementation
def hardswish_manual(x):
    return x * F.relu6(x + 3) / 6

# Common use in mobile architectures
class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hardswish = nn.Hardswish()
    
    def forward(self, x):
        return self.hardswish(self.bn(self.conv(x)))`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Piecewise smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-mobile-alt',
                title: 'Mobile optimized',
                description: 'Designed for efficient inference on mobile devices'
            },
            {
                icon: 'fas fa-tachometer-alt',
                title: 'Faster than Swish',
                description: 'Linear approximation is much more efficient'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Good approximation',
                description: 'Closely approximates Swish behavior'
            },
            {
                icon: 'fas fa-microchip',
                title: 'Hardware friendly',
                description: 'Uses only basic arithmetic operations'
            }
        ]
    },
    {
        id: 'celu',
        name: 'CELU',
        fullName: 'Continuously Differentiable ELU',
        description: 'A continuously differentiable variant of ELU that provides smooth gradients throughout the function domain.',
        formula: 'f(x) = max(0, x) + min(0, α(e^(x/α) - 1))',
        category: 'advanced',
        tags: ['smooth', 'differentiable', 'ELU-variant'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.CELU()
celu = nn.CELU(alpha=1.0)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = celu(x)
print(output)

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.celu(x, alpha=1.0)

# Custom alpha parameter
celu_custom = nn.CELU(alpha=0.5)
output_custom = celu_custom(x)`,
        properties: {
            'Output Range': '(-α, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'Nearly'
        },
        checklist: [
            {
                icon: 'fas fa-wave-square',
                title: 'Continuously differentiable',
                description: 'Smooth gradients everywhere prevent optimization issues'
            },
            {
                icon: 'fas fa-balance-scale',
                title: 'Better than ELU',
                description: 'Improved gradient flow compared to standard ELU'
            },
            {
                icon: 'fas fa-calculator',
                title: 'Computationally expensive',
                description: 'Exponential calculation makes it slower than ReLU'
            },
            {
                icon: 'fas fa-sliders-h',
                title: 'Tunable parameter',
                description: 'Alpha parameter allows customization of negative region'
            }
        ]
    },
    {
        id: 'hardshrink',
        name: 'Hardshrink',
        fullName: 'Hard Shrinkage Function',
        description: 'Zeros out values within a threshold range while preserving values outside, promoting sparsity.',
        formula: 'f(x) = x if |x| > λ, 0 otherwise',
        category: 'advanced',
        tags: ['sparsity', 'thresholding', 'regularization'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Hardshrink()
hardshrink = nn.Hardshrink(lambd=0.5)
x = torch.tensor([-2.0, -0.3, 0.0, 0.3, 2.0])
output = hardshrink(x)
print(output)  # tensor([-2.0000, 0.0000, 0.0000, 0.0000, 2.0000])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.hardshrink(x, lambd=0.5)

# Example in sparse coding
class SparseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.hardshrink = nn.Hardshrink(lambd=threshold)
    
    def forward(self, x):
        return self.hardshrink(self.encoder(x))`,
        properties: {
            'Output Range': '(-∞, -λ] ∪ {0} ∪ [λ, ∞)',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Piecewise',
            'Zero-centered': 'Yes'
        },
        checklist: [
            {
                icon: 'fas fa-filter',
                title: 'Promotes sparsity',
                description: 'Zeros small values, creating sparse representations'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Non-differentiable',
                description: 'Discontinuous at threshold points can affect gradients'
            },
            {
                icon: 'fas fa-sliders-h',
                title: 'Threshold parameter',
                description: 'Lambda controls the sparsity level'
            },
            {
                icon: 'fas fa-compress',
                title: 'Feature selection',
                description: 'Useful for automatic feature selection and denoising'
            }
        ]
    },
    {
        id: 'hardsigmoid',
        name: 'Hardsigmoid',
        fullName: 'Hard Sigmoid',
        description: 'Piecewise linear approximation of sigmoid that\'s computationally efficient for mobile devices.',
        formula: 'f(x) = max(0, min(1, (x + 3)/6))',
        category: 'advanced',
        tags: ['efficient', 'mobile', 'sigmoid-approximation'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Hardsigmoid()
hardsigmoid = nn.Hardsigmoid()
x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
output = hardsigmoid(x)
print(output)  # tensor([0.0000, 0.3333, 0.5000, 0.6667, 1.0000])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.hardsigmoid(x)

# Common use in mobile architectures
class MobileGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.hardsigmoid = nn.Hardsigmoid()
    
    def forward(self, x):
        gate = self.hardsigmoid(self.conv(x))
        return x * gate`,
        properties: {
            'Output Range': '[0, 1]',
            'Smoothness': 'Piecewise linear',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-mobile-alt',
                title: 'Mobile optimized',
                description: 'Much faster than sigmoid for mobile deployment'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Good approximation',
                description: 'Closely approximates sigmoid in useful range'
            },
            {
                icon: 'fas fa-door-open',
                title: 'Perfect for gates',
                description: 'Excellent choice for attention and gating mechanisms'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Limited expressiveness',
                description: 'Linear approximation may hurt performance vs true sigmoid'
            }
        ]
    },
    {
        id: 'hardtanh',
        name: 'Hardtanh',
        fullName: 'Hard Hyperbolic Tangent',
        description: 'Piecewise linear approximation of tanh with configurable bounds, efficient for mobile deployment.',
        formula: 'f(x) = max(min_val, min(max_val, x))',
        category: 'advanced',
        tags: ['efficient', 'bounded', 'tanh-approximation'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Hardtanh()
hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
output = hardtanh(x)
print(output)  # tensor([-1., -1., 0., 1., 1.])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.hardtanh(x, min_val=-1.0, max_val=1.0)

# Custom bounds
hardtanh_custom = nn.Hardtanh(min_val=-2.0, max_val=2.0)

# Common use in RNNs
class EfficientRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.hardtanh = nn.Hardtanh()
    
    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        return self.hardtanh(self.linear(combined))`,
        properties: {
            'Output Range': '[min_val, max_val]',
            'Smoothness': 'Piecewise linear',
            'Monotonic': 'Yes (in bounds)',
            'Zero-centered': 'Yes'
        },
        checklist: [
            {
                icon: 'fas fa-tachometer-alt',
                title: 'Very fast computation',
                description: 'Simple clipping operation, no exponentials'
            },
            {
                icon: 'fas fa-balance-scale',
                title: 'Zero-centered and bounded',
                description: 'Combines benefits of bounded output with zero centering'
            },
            {
                icon: 'fas fa-mobile-alt',
                title: 'Mobile friendly',
                description: 'Excellent for resource-constrained environments'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Saturates easily',
                description: 'May limit gradient flow for large inputs'
            }
        ]
    },
    {
        id: 'logsigmoid',
        name: 'LogSigmoid',
        fullName: 'Log Sigmoid',
        description: 'Computes the logarithm of sigmoid in a numerically stable way, commonly used with NLL loss.',
        formula: 'f(x) = log(1/(1 + e^(-x))) = log(sigmoid(x))',
        category: 'common',
        tags: ['numerically-stable', 'probability', 'log-space'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.LogSigmoid()
logsigmoid = nn.LogSigmoid()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = logsigmoid(x)
print(output)  # tensor([-2.1269, -1.3133, -0.6931, -0.3133, -0.1269])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.logsigmoid(x)

# Common use with NLL loss
class BinaryClassifierNLL(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.logsigmoid = nn.LogSigmoid()
    
    def forward(self, x):
        logits = self.linear(x)
        return self.logsigmoid(logits)
        
# Use with NLLLoss for binary classification
criterion = nn.NLLLoss()`,
        properties: {
            'Output Range': '(-∞, 0]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-calculator',
                title: 'Numerically stable',
                description: 'Avoids numerical issues when computing log(sigmoid(x))'
            },
            {
                icon: 'fas fa-percentage',
                title: 'Log probabilities',
                description: 'Perfect for models requiring log probability outputs'
            },
            {
                icon: 'fas fa-link',
                title: 'Works with NLL loss',
                description: 'Designed to work seamlessly with NLLLoss'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Always negative',
                description: 'Outputs are always negative (log of values in [0,1])'
            }
        ]
    },
    {
        id: 'relu6',
        name: 'ReLU6',
        fullName: 'ReLU Clamped at 6',
        description: 'ReLU with upper bound at 6, preventing activation explosion while maintaining computational benefits.',
        formula: 'f(x) = min(max(0, x), 6)',
        category: 'common',
        tags: ['bounded', 'mobile', 'quantization-friendly'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.ReLU6()
relu6 = nn.ReLU6()
x = torch.tensor([-2.0, 0.0, 3.0, 6.0, 8.0])
output = relu6(x)
print(output)  # tensor([0., 0., 3., 6., 6.])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.relu6(x)

# Common use in mobile networks (MobileNet)
class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6()
    
    def forward(self, x):
        return self.relu6(self.bn(self.conv(x)))`,
        properties: {
            'Output Range': '[0, 6]',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Yes (in range)',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-mobile-alt',
                title: 'Mobile optimized',
                description: 'Designed specifically for mobile neural networks'
            },
            {
                icon: 'fas fa-microchip',
                title: 'Quantization friendly',
                description: 'Bounded output works well with 8-bit quantization'
            },
            {
                icon: 'fas fa-bolt',
                title: 'Fast like ReLU',
                description: 'Simple clamping operation with ReLU speed'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Can limit expressiveness',
                description: 'Upper bound may constrain model capacity'
            }
        ]
    },
    {
        id: 'rrelu',
        name: 'RReLU',
        fullName: 'Randomized Leaky ReLU',
        description: 'LeakyReLU with random negative slope during training to reduce overfitting.',
        formula: 'f(x) = max(ax, x) where a ~ U(lower, upper)',
        category: 'advanced',
        tags: ['randomized', 'regularization', 'anti-overfitting'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.RReLU()
rrelu = nn.RReLU(lower=1./8, upper=1./3)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = rrelu(x)
print(output)  # Different each time in training mode

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.rrelu(x, lower=1./8, upper=1./3, training=True)

# Example in regularized network
class RegularizedNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.rrelu1 = nn.RReLU(lower=0.1, upper=0.3)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.rrelu2 = nn.RReLU(lower=0.1, upper=0.3)
    
    def forward(self, x):
        x = self.rrelu1(self.linear1(x))
        x = self.rrelu2(self.linear2(x))
        return x`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-dice',
                title: 'Built-in regularization',
                description: 'Random slope acts as regularization during training'
            },
            {
                icon: 'fas fa-heartbeat',
                title: 'Prevents dead neurons',
                description: 'Non-zero gradient for negative inputs keeps neurons alive'
            },
            {
                icon: 'fas fa-balance-scale',
                title: 'Train/eval difference',
                description: 'Uses random slope in training, fixed average in evaluation'
            },
            {
                icon: 'fas fa-sliders-h',
                title: 'Hyperparameter sensitive',
                description: 'Lower and upper bounds need careful tuning'
            }
        ]
    },
    {
        id: 'softplus',
        name: 'Softplus',
        fullName: 'Soft Plus',
        description: 'Smooth approximation of ReLU that\'s always positive, useful for variance parameters in probabilistic models.',
        formula: 'f(x) = (1/β) * log(1 + e^(βx))',
        category: 'smooth',
        tags: ['smooth', 'positive', 'probabilistic'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Softplus()
softplus = nn.Softplus(beta=1, threshold=20)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = softplus(x)
print(output)  # tensor([0.1269, 0.3133, 0.6931, 1.3133, 2.1269])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.softplus(x, beta=1, threshold=20)

# Common use in VAEs for variance
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim * 2)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        h = self.linear(x)
        mu, log_var = h.chunk(2, dim=-1)
        # Use softplus to ensure positive variance
        var = self.softplus(log_var)
        return mu, var`,
        properties: {
            'Output Range': '(0, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth ReLU alternative',
                description: 'Provides smooth gradients everywhere, no dead neurons'
            },
            {
                icon: 'fas fa-plus',
                title: 'Always positive',
                description: 'Perfect for parameters that must be positive (e.g., variance)'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Good for probabilistic models',
                description: 'Natural choice for positive constraints in VAEs, GANs'
            },
            {
                icon: 'fas fa-calculator',
                title: 'More expensive than ReLU',
                description: 'Logarithm and exponential make it computationally costly'
            }
        ]
    },
    {
        id: 'softshrink',
        name: 'Softshrink',
        fullName: 'Soft Shrinkage Function',
        description: 'Applies soft thresholding that shrinks values toward zero, promoting sparsity.',
        formula: 'f(x) = sign(x) * max(0, |x| - λ)',
        category: 'advanced',
        tags: ['sparsity', 'soft-thresholding', 'shrinkage'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Softshrink()
softshrink = nn.Softshrink(lambd=0.5)
x = torch.tensor([-2.0, -0.3, 0.0, 0.3, 2.0])
output = softshrink(x)
print(output)  # tensor([-1.5000, 0.0000, 0.0000, 0.0000, 1.5000])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.softshrink(x, lambd=0.5)

# Example in sparse representation learning
class SoftSparseLayer(nn.Module):
    def __init__(self, dim, threshold=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.softshrink = nn.Softshrink(lambd=threshold)
    
    def forward(self, x):
        return self.softshrink(self.linear(x))`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Piecewise',
            'Zero-centered': 'Yes'
        },
        checklist: [
            {
                icon: 'fas fa-compress',
                title: 'Soft sparsity',
                description: 'Gradually shrinks small values toward zero'
            },
            {
                icon: 'fas fa-balance-scale',
                title: 'Zero-centered',
                description: 'Symmetric around zero, good for hidden layers'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Non-differentiable at ±λ',
                description: 'Gradient issues at threshold points'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Feature selection',
                description: 'Useful for automatic feature selection tasks'
            }
        ]
    },
    {
        id: 'softsign',
        name: 'Softsign',
        fullName: 'Soft Sign Function',
        description: 'Alternative to tanh with polynomial decay, bounded output without exponentials.',
        formula: 'f(x) = x / (1 + |x|)',
        category: 'smooth',
        tags: ['polynomial', 'bounded', 'tanh-alternative'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Softsign()
softsign = nn.Softsign()
x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
output = softsign(x)
print(output)  # tensor([-0.9091, -0.5000, 0.0000, 0.5000, 0.9091])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.softsign(x)

# Example in neural network
class SoftsignNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.softsign1 = nn.Softsign()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.softsign2 = nn.Softsign()
    
    def forward(self, x):
        x = self.softsign1(self.linear1(x))
        x = self.softsign2(self.linear2(x))
        return x`,
        properties: {
            'Output Range': '(-1, 1)',
            'Smoothness': 'Smooth (except at 0)',
            'Monotonic': 'Yes',
            'Zero-centered': 'Yes'
        },
        checklist: [
            {
                icon: 'fas fa-balance-scale',
                title: 'Zero-centered and bounded',
                description: 'Good alternative to tanh without exponentials'
            },
            {
                icon: 'fas fa-chart-line',
                title: 'Slower saturation',
                description: 'Polynomial decay saturates more slowly than tanh'
            },
            {
                icon: 'fas fa-tachometer-alt',
                title: 'No exponentials',
                description: 'Computationally simpler than tanh'
            },
            {
                icon: 'fas fa-question-circle',
                title: 'Less common',
                description: 'Limited research and practical usage compared to tanh'
            }
        ]
    },
    {
        id: 'tanhshrink',
        name: 'Tanhshrink',
        fullName: 'Tanh Shrinkage',
        description: 'Computes the difference between input and its tanh, creating unique shrinkage properties.',
        formula: 'f(x) = x - tanh(x)',
        category: 'advanced',
        tags: ['shrinkage', 'mathematical', 'specialized'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Tanhshrink()
tanhshrink = nn.Tanhshrink()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tanhshrink(x)
print(output)  # tensor([-0.0360, -0.2384, 0.0000, 0.2384, 1.0360])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.tanhshrink(x)

# Manual implementation equivalent
def tanhshrink_manual(x):
    return x - torch.tanh(x)

# Specialized usage example
class TanhshrinkLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.tanhshrink = nn.Tanhshrink()
    
    def forward(self, x):
        return self.tanhshrink(self.linear(x))`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'Yes'
        },
        checklist: [
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth everywhere',
                description: 'Infinitely differentiable across all inputs'
            },
            {
                icon: 'fas fa-question-circle',
                title: 'Specialized use case',
                description: 'Limited practical applications, mostly research'
            },
            {
                icon: 'fas fa-balance-scale',
                title: 'Zero-centered',
                description: 'Symmetric behavior around zero'
            },
            {
                icon: 'fas fa-calculator',
                title: 'Computationally expensive',
                description: 'Requires hyperbolic tangent computation'
            }
        ]
    },
    {
        id: 'threshold',
        name: 'Threshold',
        fullName: 'Threshold Function',
        description: 'Hard threshold that replaces values below threshold with a specified value.',
        formula: 'f(x) = x if x > threshold, value otherwise',
        category: 'advanced',
        tags: ['hard-threshold', 'binary', 'step-function'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Threshold()
threshold = nn.Threshold(threshold=0.1, value=20)
x = torch.tensor([-1.0, 0.05, 0.1, 0.5, 1.0])
output = threshold(x)
print(output)  # tensor([20.0000, 20.0000, 20.0000, 0.5000, 1.0000])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.threshold(x, threshold=0.1, value=20, inplace=False)

# Custom thresholding
threshold_custom = nn.Threshold(threshold=0.5, value=0.0)

# Example for creating binary activations
class BinaryGate(nn.Module):
    def __init__(self, threshold_val=0.0):
        super().__init__()
        self.threshold = nn.Threshold(threshold=threshold_val, value=0.0)
    
    def forward(self, x):
        return self.threshold(x)`,
        properties: {
            'Output Range': '{value} ∪ (threshold, ∞)',
            'Smoothness': 'Non-smooth',
            'Monotonic': 'Piecewise',
            'Zero-centered': 'Depends on params'
        },
        checklist: [
            {
                icon: 'fas fa-cut',
                title: 'Hard decision boundary',
                description: 'Creates sharp cutoff at threshold value'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Non-differentiable',
                description: 'Discontinuous function can block gradients'
            },
            {
                icon: 'fas fa-sliders-h',
                title: 'Highly configurable',
                description: 'Both threshold and replacement value are tunable'
            },
            {
                icon: 'fas fa-microchip',
                title: 'Simple operation',
                description: 'Very fast comparison and assignment'
            }
        ]
    },
    {
        id: 'logsoftmax',
        name: 'LogSoftmax',
        fullName: 'Log Softmax',
        description: 'Computes logarithm of softmax in numerically stable way, commonly used with NLL loss for classification.',
        formula: 'f(x_i) = x_i - log(Σ_j e^(x_j))',
        category: 'common',
        tags: ['numerically-stable', 'classification', 'log-probabilities'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.LogSoftmax()
log_softmax = nn.LogSoftmax(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
output = log_softmax(x)
print(output)

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.log_softmax(x, dim=1)

# Common use in classification
class ClassifierWithLogSoftmax(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        logits = self.linear(x)
        return self.log_softmax(logits)

# Use with NLLLoss
criterion = nn.NLLLoss()`,
        properties: {
            'Output Range': '(-∞, 0]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Order-preserving',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-calculator',
                title: 'Numerically stable',
                description: 'Avoids numerical overflow in softmax computation'
            },
            {
                icon: 'fas fa-link',
                title: 'Perfect with NLL loss',
                description: 'Designed specifically for use with NLLLoss'
            },
            {
                icon: 'fas fa-percentage',
                title: 'Log probabilities',
                description: 'Outputs log probabilities that sum to log(1) = 0'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Always negative',
                description: 'All outputs are negative (logarithm of probabilities)'
            }
        ]
    },
    {
        id: 'softmax2d',
        name: 'Softmax2d',
        fullName: 'Softmax 2D',
        description: 'Applies softmax over features at each spatial location, useful for dense prediction tasks.',
        formula: 'Applied per (h,w): softmax over channel dimension',
        category: 'advanced',
        tags: ['spatial', 'dense-prediction', 'segmentation'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Softmax2d()
softmax2d = nn.Softmax2d()
# Input: (batch_size, channels, height, width)
x = torch.randn(2, 3, 4, 4)
output = softmax2d(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Sum over channels at (0,0): {output[0, :, 0, 0].sum()}")

# Equivalent using regular softmax
output_equiv = F.softmax(x, dim=1)

# Common use in semantic segmentation
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)
        self.softmax2d = nn.Softmax2d()
    
    def forward(self, x):
        logits = self.conv(x)
        return self.softmax2d(logits)`,
        properties: {
            'Output Range': '[0, 1]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Order-preserving',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-map',
                title: 'Spatial probability maps',
                description: 'Creates probability distribution per pixel location'
            },
            {
                icon: 'fas fa-eye',
                title: 'Perfect for segmentation',
                description: 'Ideal for semantic/instance segmentation tasks'
            },
            {
                icon: 'fas fa-layer-group',
                title: 'Preserves spatial structure',
                description: 'Maintains spatial relationships while normalizing'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Can wash out small activations',
                description: 'May suppress important but small feature responses'
            }
        ]
    },
    {
        id: 'softmin',
        name: 'Softmin',
        fullName: 'Soft Minimum',
        description: 'Assigns higher probabilities to smaller values, inverse of softmax behavior.',
        formula: 'f(x_i) = e^(-x_i) / Σ_j e^(-x_j)',
        category: 'advanced',
        tags: ['inverse-softmax', 'minimum-emphasis', 'attention'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.Softmin()
softmin = nn.Softmin(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
output = softmin(x)
print(output)  # Smaller values get higher probabilities

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.softmin(x, dim=1)

# Equivalent to softmax of negative input
output_equiv = F.softmax(-x, dim=1)

# Example in distance-based attention
class DistanceAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.softmin = nn.Softmin(dim=1)
    
    def forward(self, query, keys):
        # Compute distances
        distances = torch.norm(query.unsqueeze(1) - keys, dim=2)
        # Apply softmin to emphasize closer items
        attention_weights = self.softmin(distances)
        return attention_weights`,
        properties: {
            'Output Range': '[0, 1]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Inversely monotonic',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-arrow-down',
                title: 'Emphasizes smaller values',
                description: 'Higher probability for smaller input values'
            },
            {
                icon: 'fas fa-eye',
                title: 'Useful for distance metrics',
                description: 'Good for attention mechanisms based on distance'
            },
            {
                icon: 'fas fa-question-circle',
                title: 'Limited use cases',
                description: 'Specialized function with narrow applications'
            },
            {
                icon: 'fas fa-calculator',
                title: 'Same cost as softmax',
                description: 'Computational complexity similar to softmax'
            }
        ]
    },
    {
        id: 'adaptivelogsoftmax',
        name: 'AdaptiveLogSoftmax',
        fullName: 'Adaptive Log Softmax With Loss',
        description: 'Efficient softmax approximation for large vocabularies using hierarchical clustering.',
        formula: 'Hierarchical softmax with adaptive clustering',
        category: 'advanced',
        tags: ['large-vocabulary', 'efficient', 'hierarchical'],
        pytorch_code: `import torch
import torch.nn as nn

# AdaptiveLogSoftmaxWithLoss for large vocabularies
vocab_size = 50000
input_size = 512
cutoffs = [2000, 10000]  # Hierarchical cutoffs

adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
    in_features=input_size,
    n_classes=vocab_size,
    cutoffs=cutoffs,
    div_value=4.0
)

# Input: (seq_len, batch_size, input_size)
# Target: (seq_len * batch_size,)
input_tensor = torch.randn(10, 32, input_size)
target = torch.randint(0, vocab_size, (10 * 32,))

# Forward pass returns output and loss
output, loss = adaptive_softmax(input_tensor.view(-1, input_size), target)

# For inference, use log_prob method
log_probs = adaptive_softmax.log_prob(input_tensor.view(-1, input_size))`,
        properties: {
            'Output Range': '(-∞, 0]',
            'Smoothness': 'Smooth',
            'Monotonic': 'Order-preserving',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-tachometer-alt',
                title: 'Efficient for large vocabularies',
                description: 'O(√V) complexity instead of O(V) for vocabulary size V'
            },
            {
                icon: 'fas fa-layer-group',
                title: 'Hierarchical structure',
                description: 'Groups frequent and rare words for efficient computation'
            },
            {
                icon: 'fas fa-comment',
                title: 'Perfect for language models',
                description: 'Designed specifically for large vocabulary NLP tasks'
            },
            {
                icon: 'fas fa-cogs',
                title: 'Complex setup',
                description: 'Requires careful cutoff selection and vocabulary analysis'
            }
        ]
    },
    {
        id: 'multiheadattention',
        name: 'MultiheadAttn',
        fullName: 'Multi-Head Attention',
        description: 'Attention mechanism that allows the model to attend to information from different representation subspaces.',
        formula: 'MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O',
        category: 'advanced',
        tags: ['attention', 'transformer', 'multi-head'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.MultiheadAttention()
embed_dim = 512
num_heads = 8
multihead_attn = nn.MultiheadAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    dropout=0.1,
    batch_first=True
)

# Input tensors
seq_len, batch_size = 20, 32
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

# Forward pass
attn_output, attn_weights = multihead_attn(query, key, value)

# Self-attention (Q, K, V are the same)
self_attn_output, self_attn_weights = multihead_attn(query, query, query)

# With attention mask
attn_mask = torch.tril(torch.ones(seq_len, seq_len))  # Causal mask
masked_output, _ = multihead_attn(query, key, value, attn_mask=attn_mask)`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'No',
            'Zero-centered': 'Depends on input'
        },
        checklist: [
            {
                icon: 'fas fa-robot',
                title: 'Transformer core',
                description: 'Essential component of transformer architectures'
            },
            {
                icon: 'fas fa-eye',
                title: 'Attention mechanism',
                description: 'Allows model to focus on relevant parts of input'
            },
            {
                icon: 'fas fa-layer-group',
                title: 'Multiple representation subspaces',
                description: 'Each head learns different types of relationships'
            },
            {
                icon: 'fas fa-memory',
                title: 'Memory intensive',
                description: 'Quadratic memory complexity in sequence length'
            }
        ]
    },
    {
        id: 'silu',
        name: 'SiLU',
        fullName: 'Sigmoid Linear Unit',
        description: 'Standalone SiLU activation, identical to Swish but with separate PyTorch implementation.',
        formula: 'f(x) = x * σ(x) = x / (1 + e^(-x))',
        category: 'advanced',
        tags: ['smooth', 'self-gating', 'swish-equivalent'],
        pytorch_code: `import torch
import torch.nn as nn

# Method 1: Using nn.SiLU() (recommended)
silu = nn.SiLU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = silu(x)
print(output)  # tensor([-0.2384, -0.2689, 0.0000, 0.7311, 1.7616])

# Method 2: Using functional interface
import torch.nn.functional as F
output = F.silu(x)

# Note: SiLU and Swish are mathematically identical
# SiLU is the official PyTorch name for this activation

# Example usage
class SiLUNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        return self.linear2(self.silu(self.linear1(x)))`,
        properties: {
            'Output Range': '(-∞, ∞)',
            'Smoothness': 'Smooth',
            'Monotonic': 'Yes',
            'Zero-centered': 'No'
        },
        checklist: [
            {
                icon: 'fas fa-wave-square',
                title: 'Smooth everywhere',
                description: 'Differentiable across entire input range'
            },
            {
                icon: 'fas fa-door-open',
                title: 'Self-gating property',
                description: 'Acts as its own gate through sigmoid multiplication'
            },
            {
                icon: 'fas fa-star',
                title: 'Official PyTorch name',
                description: 'Preferred name over Swish in PyTorch ecosystem'
            },
            {
                icon: 'fas fa-robot',
                title: 'Modern architectures',
                description: 'Used in many state-of-the-art models'
            }
        ]
    }
];

class ActivationExplorer {
    constructor() {
        this.allFunctions = activationFunctions;
        this.filteredFunctions = activationFunctions;
        this.currentFilter = 'all';
        this.searchTerm = '';
        this.currentFramework = 'pytorch';
        
        this.initializeElements();
        this.setupEventListeners();
        this.renderCards();
    }
    
    initializeElements() {
        this.searchInput = document.getElementById('searchInput');
        this.clearSearch = document.getElementById('clearSearch');
        this.cardsGrid = document.getElementById('cardsGrid');
        this.modal = document.getElementById('modal');
        this.modalClose = document.getElementById('modalClose');
        this.noResults = document.getElementById('noResults');
        this.filterTabs = document.querySelectorAll('.filter-tab');
        this.frameworkTabs = document.querySelectorAll('.framework-tab');
    }
    
    setupEventListeners() {
        this.searchInput.addEventListener('input', (e) => {
            this.searchTerm = e.target.value.toLowerCase();
            this.updateClearButton();
            this.filterAndRender();
        });
        
        this.clearSearch.addEventListener('click', () => {
            this.searchInput.value = '';
            this.searchTerm = '';
            this.updateClearButton();
            this.filterAndRender();
        });
        
        this.filterTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.setActiveFilter(e.target.dataset.category);
                this.filterAndRender();
            });
        });
        
        this.modalClose.addEventListener('click', () => {
            this.closeModal();
        });
        
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeModal();
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
        
        this.frameworkTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.setActiveFramework(e.target.dataset.framework);
                this.updateModalCode();
            });
        });

        document.getElementById('copyCode').addEventListener('click', () => {
            const code = document.getElementById('modalCode').textContent;
            navigator.clipboard.writeText(code).then(() => {
                const btn = document.getElementById('copyCode');
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                }, 1000);
            });
        });
    }
    
    updateClearButton() {
        if (this.searchTerm) {
            this.clearSearch.classList.add('visible');
        } else {
            this.clearSearch.classList.remove('visible');
        }
    }
    
    setActiveFilter(category) {
        this.currentFilter = category;
        this.filterTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.category === category);
        });
    }

    setActiveFramework(framework) {
        this.currentFramework = framework;
        this.frameworkTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.framework === framework);
        });
    }

    updateModalCode() {
        const func = this.currentModalFunction;
        if (!func) return;

        const codeKey = this.currentFramework === 'pytorch' ? 'pytorch_code' : 'tensorflow_code';
        let code = func[codeKey];
        
        // If no TensorFlow code exists, use the mapping
        if (!code && this.currentFramework === 'tensorflow') {
            code = this.getTensorFlowCode(func.id);
        }

        document.getElementById('modalCode').textContent = code || 'Implementation not available';
    }
    
    filterAndRender() {
        this.filteredFunctions = this.allFunctions.filter(func => {
            const matchesSearch = !this.searchTerm || 
                func.name.toLowerCase().includes(this.searchTerm) ||
                func.fullName.toLowerCase().includes(this.searchTerm) ||
                func.description.toLowerCase().includes(this.searchTerm) ||
                func.tags.some(tag => tag.toLowerCase().includes(this.searchTerm));
            
            const matchesFilter = this.currentFilter === 'all' || 
                func.category === this.currentFilter ||
                (this.currentFilter === 'smooth' && func.tags.includes('smooth'));
            
            return matchesSearch && matchesFilter;
        });
        
        this.renderCards();
    }
    
    renderCards() {
        if (this.filteredFunctions.length === 0) {
            this.cardsGrid.style.display = 'none';
            this.noResults.style.display = 'block';
            return;
        }
        
        this.cardsGrid.style.display = 'grid';
        this.noResults.style.display = 'none';
        
        this.cardsGrid.innerHTML = this.filteredFunctions.map(func => `
            <div class="card fade-in" onclick="explorer.openModal('${func.id}')">
                <div class="card-header">
                    <h3 class="card-title">${func.name}</h3>
                    <p class="card-subtitle">${func.fullName}</p>
                </div>
                <p class="card-description">${func.description}</p>
                <div class="card-tags">
                    ${func.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                <div class="card-properties">
                    <div class="property">
                        <i class="fas fa-chart-line property-icon"></i>
                        <span>Range: ${func.properties['Output Range']}</span>
                    </div>
                    <div class="property">
                        <i class="fas fa-wave-square property-icon"></i>
                        <span>${func.properties['Smoothness']}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    getTensorFlowCode(functionId) {
        const tensorflowCodes = {
            relu: `import tensorflow as tf

# Method 1: Using tf.keras.activations.relu
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.relu(x)
print(output)  # tf.Tensor([0. 0. 0. 1. 2.], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.relu
output = tf.nn.relu(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu')
])

# Method 4: As separate layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.ReLU()
])`,

            sigmoid: `import tensorflow as tf

# Method 1: Using tf.keras.activations.sigmoid
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.sigmoid(x)
print(output)  # tf.Tensor([0.1192 0.2689 0.5000 0.7311 0.8808], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.sigmoid
output = tf.nn.sigmoid(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Method 4: Binary classification
class BinaryClassifier(tf.keras.Model):
    def __init__(self, input_size):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        return self.dense(x)`,

            tanh: `import tensorflow as tf

# Method 1: Using tf.keras.activations.tanh
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.tanh(x)
print(output)  # tf.Tensor([-0.964 -0.7616 0. 0.7616 0.964], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.tanh
output = tf.nn.tanh(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh')
])

# Method 4: RNN example
class SimpleRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.i2h = tf.keras.layers.Dense(hidden_size)
        self.h2h = tf.keras.layers.Dense(hidden_size, use_bias=False)
    
    def call(self, x, hidden):
        return tf.nn.tanh(self.i2h(x) + self.h2h(hidden))`,

            leakyrelu: `import tensorflow as tf

# Method 1: Using tf.keras.layers.LeakyReLU
leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = leaky_relu(x)
print(output)  # tf.Tensor([-0.02 -0.01 0. 1. 2.], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.leaky_relu
output = tf.nn.leaky_relu(x, alpha=0.01)

# Method 3: As layer in model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.LeakyReLU(alpha=0.01)
])`,

            gelu: `import tensorflow as tf

# Method 1: Using tf.keras.activations.gelu
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.gelu(x)
print(output)  # tf.Tensor([-0.0454 -0.1587 0. 0.8413 1.9545], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.gelu
output = tf.nn.gelu(x)

# Method 3: As activation in Dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='gelu')
])

# Method 4: Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(d_ff, activation='gelu')
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(0.1)
    
    def call(self, x, training=None):
        return self.linear2(self.dropout(self.linear1(x), training=training))`,

            elu: `import tensorflow as tf

# Method 1: Using tf.keras.layers.ELU
elu = tf.keras.layers.ELU(alpha=1.0)
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = elu(x)
print(output)  # tf.Tensor([-0.8647 -0.6321 0. 1. 2.], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.elu
output = tf.nn.elu(x)

# Method 3: As layer activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.ELU(alpha=1.0)
])`,

            selu: `import tensorflow as tf

# Method 1: Using tf.keras.activations.selu
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.selu(x)
print(output)  # tf.Tensor([-0.9088 -0.664 0. 1.0507 2.1014], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.selu
output = tf.nn.selu(x)

# Method 3: SELU network with proper initialization
class SELUNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layers = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='selu',
                                kernel_initializer='lecun_normal'),
            tf.keras.layers.AlphaDropout(0.1),
            tf.keras.layers.Dense(num_classes)
        ])
    
    def call(self, x):
        return self.layers(x)`,

            glu: `import tensorflow as tf

# Custom GLU implementation (TensorFlow doesn't have native GLU)
class GLU(tf.keras.layers.Layer):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def call(self, x):
        a, b = tf.split(x, 2, axis=self.dim)
        return a * tf.keras.activations.sigmoid(b)

# Usage
x = tf.random.normal((2, 8))  # Batch size 2, 8 features
glu = GLU(dim=-1)
output = glu(x)  # Output: (2, 4)

# In feed-forward layer
class GLUFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear = tf.keras.layers.Dense(d_ff * 2)
        self.glu = GLU(dim=-1)
    
    def call(self, x):
        return self.glu(self.linear(x))`,

            prelu: `import tensorflow as tf

# Method 1: Using tf.keras.layers.PReLU
prelu = tf.keras.layers.PReLU()
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = prelu(x)
print(f"Output: {output}")
print(f"Learned parameter: {prelu.alpha.numpy()}")

# Method 2: Channel-wise parameters
prelu_channelwise = tf.keras.layers.PReLU(shared_axes=None)

# Method 3: Network example
class PReLUNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size)
        self.prelu1 = tf.keras.layers.PReLU()
        self.dense2 = tf.keras.layers.Dense(hidden_size)
        self.prelu2 = tf.keras.layers.PReLU()
    
    def call(self, x):
        x = self.prelu1(self.dense1(x))
        return self.prelu2(self.dense2(x))`,

            swish: `import tensorflow as tf

# Method 1: Using tf.keras.activations.swish (SiLU in PyTorch)
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.swish(x)
print(output)  # tf.Tensor([-0.2384 -0.2689 0. 0.7311 1.7616], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.swish
output = tf.nn.swish(x)

# Method 3: As activation function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='swish')
])

# Method 4: Modern architecture
class ModernBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(dim, activation='swish')
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        return self.norm(self.dense(x))`,

            silu: `import tensorflow as tf

# Note: SiLU is called Swish in TensorFlow
# Method 1: Using tf.keras.activations.swish
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.swish(x)  # SiLU = Swish
print(output)  # tf.Tensor([-0.2384 -0.2689 0. 0.7311 1.7616], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.swish
output = tf.nn.swish(x)

# Method 3: Manual SiLU implementation
def silu(x):
    return x * tf.keras.activations.sigmoid(x)

# Method 4: As activation function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='swish')  # SiLU is 'swish' in TF
])`,
            
            mish: `import tensorflow as tf

# TensorFlow doesn't have native Mish, here's custom implementation
def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))

class Mish(tf.keras.layers.Layer):
    def call(self, x):
        return x * tf.nn.tanh(tf.nn.softplus(x))

# Usage
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
mish_layer = Mish()
output = mish_layer(x)
print(output)  # Custom implementation

# In CNN
class MishCNN(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same')
        self.mish1 = Mish()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.mish2 = Mish()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)
    
    def call(self, x):
        x = self.mish1(self.conv1(x))
        x = self.mish2(self.conv2(x))
        x = self.pool(x)
        return self.fc(x)`,

            softmax: `import tensorflow as tf

# Method 1: Using tf.keras.activations.softmax
x = tf.constant([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
output = tf.keras.activations.softmax(x)
print(output)  # Each row sums to 1.0

# Method 2: Using tf.nn.softmax
output = tf.nn.softmax(x, axis=-1)

# Method 3: As activation in Dense layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='softmax')
])

# Method 4: Classification model
class Classifier(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense = tf.keras.layers.Dense(num_classes)
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, x, training=None):
        logits = self.dense(x)
        if not training:
            return self.softmax(logits)
        return logits`,

            hardswish: `import tensorflow as tf

# Custom Hard Swish implementation
def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0

class HardSwish(tf.keras.layers.Layer):
    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0

# Usage
x = tf.constant([-4.0, -2.0, 0.0, 2.0, 4.0])
hardswish_layer = HardSwish()
output = hardswish_layer(x)

# Mobile architecture example
class MobileBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.hardswish = HardSwish()
    
    def call(self, x, training=None):
        return self.hardswish(self.bn(self.conv(x), training=training))`,

            multiheadattention: `import tensorflow as tf

# Method 1: Using tf.keras.layers.MultiHeadAttention
embed_dim = 512
num_heads = 8
mha = tf.keras.layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=embed_dim // num_heads,
    dropout=0.1
)

# Input tensors
seq_len, batch_size = 20, 32
query = tf.random.normal((batch_size, seq_len, embed_dim))
key = tf.random.normal((batch_size, seq_len, embed_dim))
value = tf.random.normal((batch_size, seq_len, embed_dim))

# Forward pass
attn_output = mha(query, key, value)

# Self-attention
self_attn_output = mha(query, query, query)

# With attention mask
attn_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
masked_output = mha(query, key, value, attention_mask=attn_mask)

# Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        attn_output = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)`,

            logsigmoid: `import tensorflow as tf

# Method 1: Using tf.nn.log_sigmoid
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.nn.log_sigmoid(x)
print(output)  # tf.Tensor([-2.1269 -1.3133 -0.6931 -0.3133 -0.1269], shape=(5,), dtype=float32)

# Method 2: Custom layer
class LogSigmoid(tf.keras.layers.Layer):
    def call(self, x):
        return tf.nn.log_sigmoid(x)

# Method 3: Binary classification with log probabilities
class BinaryClassifierNLL(tf.keras.Model):
    def __init__(self, input_size):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, x):
        logits = self.dense(x)
        return tf.nn.log_sigmoid(logits)`,

            relu6: `import tensorflow as tf

# Method 1: Using tf.nn.relu6
x = tf.constant([-2.0, 0.0, 3.0, 6.0, 8.0])
output = tf.nn.relu6(x)
print(output)  # tf.Tensor([0. 0. 3. 6. 6.], shape=(5,), dtype=float32)

# Method 2: Using ReLU layer with max_value
relu6_layer = tf.keras.layers.ReLU(max_value=6.0)
output = relu6_layer(x)

# Method 3: MobileNet block
class MobileNetBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu6 = tf.keras.layers.ReLU(max_value=6.0)
    
    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.relu6(x)`,

            logsoftmax: `import tensorflow as tf

# Method 1: Using tf.nn.log_softmax
x = tf.constant([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]])
output = tf.nn.log_softmax(x, axis=1)
print(output)

# Method 2: Classification with log softmax
class ClassifierWithLogSoftmax(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense = tf.keras.layers.Dense(num_classes)
    
    def call(self, x):
        logits = self.dense(x)
        return tf.nn.log_softmax(logits, axis=1)

# Use with sparse categorical crossentropy for numerical stability`,

            softplus: `import tensorflow as tf

# Method 1: Using tf.keras.activations.softplus
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
output = tf.keras.activations.softplus(x)
print(output)  # tf.Tensor([0.1269 0.3133 0.6931 1.3133 2.1269], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.softplus
output = tf.nn.softplus(x)

# Method 3: In VAE for variance
class VariationalEncoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(latent_dim * 2)
    
    def call(self, x):
        h = self.dense(x)
        mu, log_var = tf.split(h, 2, axis=-1)
        var = tf.nn.softplus(log_var)  # Ensure positive
        return mu, var`,

            softsign: `import tensorflow as tf

# Method 1: Using tf.keras.activations.softsign
x = tf.constant([-10.0, -1.0, 0.0, 1.0, 10.0])
output = tf.keras.activations.softsign(x)
print(output)  # tf.Tensor([-0.9091 -0.5 0. 0.5 0.9091], shape=(5,), dtype=float32)

# Method 2: Using tf.nn.softsign
output = tf.nn.softsign(x)

# Method 3: Neural network example
class SoftsignNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='softsign')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation='softsign')
    
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)`,

            hardsigmoid: `import tensorflow as tf

# Method 1: Using tf.keras.activations.hard_sigmoid
x = tf.constant([-4.0, -1.0, 0.0, 1.0, 4.0])
output = tf.keras.activations.hard_sigmoid(x)
print(output)  # tf.Tensor([0. 0.3 0.5 0.7 1.], shape=(5,), dtype=float32)

# Note: TensorFlow's hard_sigmoid uses (x + 1) / 2, PyTorch uses (x + 3) / 6
# Method 2: PyTorch-compatible version
def hard_sigmoid_pytorch(x):
    return tf.clip_by_value((x + 3.0) / 6.0, 0.0, 1.0)

# Method 3: Mobile gate example
class MobileGate(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(channels, 1)
    
    def call(self, x):
        gate = tf.keras.activations.hard_sigmoid(self.conv(x))
        return x * gate`
        };

        return tensorflowCodes[functionId] || `# TensorFlow implementation for ${functionId} not yet available
# Please check TensorFlow documentation for equivalent functions
import tensorflow as tf

# Most activation functions have direct equivalents:
# - Use tf.keras.activations.function_name()
# - Use tf.nn.function_name()
# - Use as activation='function_name' in layers
# - Create custom layers for unsupported functions

# Example pattern:
# x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
# output = tf.keras.activations.${functionId}(x)  # if available`;
    }

    openModal(functionId) {
        const func = this.allFunctions.find(f => f.id === functionId);
        if (!func) return;
        
        this.currentModalFunction = func;
        
        document.getElementById('modalTitle').textContent = func.fullName;
        document.getElementById('modalDescription').textContent = func.description;
        document.getElementById('modalFormula').textContent = func.formula;
        
        // Update code based on current framework
        this.updateModalCode();
        
        const checklistHTML = func.checklist.map(item => `
            <div class="checklist-item">
                <i class="${item.icon} checklist-icon"></i>
                <div class="checklist-text">
                    <strong>${item.title}</strong>
                    ${item.description}
                </div>
            </div>
        `).join('');
        document.getElementById('modalChecklist').innerHTML = checklistHTML;
        
        const propertiesHTML = Object.entries(func.properties).map(([key, value]) => `
            <div class="property-card">
                <div class="property-label">${key}</div>
                <div class="property-value">${value}</div>
            </div>
        `).join('');
        document.getElementById('modalProperties').innerHTML = propertiesHTML;
        
        this.modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    closeModal() {
        this.modal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }
}

const explorer = new ActivationExplorer();