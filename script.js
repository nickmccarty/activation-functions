const activationFunctions = [
    {
        id: 'relu',
        name: 'ReLU',
        fullName: 'Rectified Linear Unit',
        description: 'The most widely used activation function that outputs the input directly if positive, otherwise outputs zero.',
        formula: 'f(x) = max(0, x)',
        category: 'common',
        tags: ['fast', 'simple', 'gradient-friendly'],
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
        code: `import torch
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
    }
];

class ActivationExplorer {
    constructor() {
        this.allFunctions = activationFunctions;
        this.filteredFunctions = activationFunctions;
        this.currentFilter = 'all';
        this.searchTerm = '';
        
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
    
    openModal(functionId) {
        const func = this.allFunctions.find(f => f.id === functionId);
        if (!func) return;
        
        document.getElementById('modalTitle').textContent = func.fullName;
        document.getElementById('modalDescription').textContent = func.description;
        document.getElementById('modalFormula').textContent = func.formula;
        document.getElementById('modalCode').textContent = func.code;
        
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