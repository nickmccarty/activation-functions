# PyTorch & TensorFlow Activation Functions Explorer

[![Demo](https://img.shields.io/badge/Demo-Live%20Site-blue?style=for-the-badge&logo=github)](https://nickmccarty.me/activation-functions)
[![PyTorch](https://img.shields.io/badge/PyTorch-Ready-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Ready-ff6f00?style=flat&logo=tensorflow)](https://tensorflow.org/)

An interactive, modern web application for exploring activation functions across **both PyTorch and TensorFlow** with detailed explanations, code examples, and practitioner guidance.

## Features

- 🔍 **Smart Search** - Find activation functions by name, description, or properties
- 🎨 **Neumorphic Design** - Modern, tactile UI with smooth animations
- 📱 **Responsive** - Works seamlessly on desktop and mobile devices
- 💡 **Interactive Cards** - Click any function for detailed information
- 📊 **Practitioner Checklists** - Actionable guidance for choosing the right activation
- 🔄 **Dual Framework Support** - Switch between PyTorch and TensorFlow implementations
- 🐍 **PyTorch Code** - Complete, runnable PyTorch examples
- 🔶 **TensorFlow Code** - Complete, runnable TensorFlow/Keras examples
- 🏷️ **Smart Filtering** - Category-based filtering (Common, Advanced, Smooth)

## Complete Activation Function Coverage ✅

### Common Activations
- **ReLU** - Rectified Linear Unit
- **Sigmoid** - Classic S-shaped activation  
- **Tanh** - Hyperbolic Tangent
- **LogSigmoid** - Numerically stable log sigmoid
- **ReLU6** - Mobile-optimized ReLU clamped at 6
- **Softmax** - Probability distribution function
- **LogSoftmax** - Numerically stable log softmax

### Advanced Activations  
- **GELU** - Gaussian Error Linear Unit (Transformer favorite)
- **ELU** - Exponential Linear Unit
- **CELU** - Continuously Differentiable ELU
- **SELU** - Scaled ELU (Self-normalizing)
- **LeakyReLU** - ReLU with small negative slope
- **RReLU** - Randomized Leaky ReLU with regularization
- **PReLU** - Parametric ReLU with learnable slope
- **SiLU/Swish** - Smooth self-gating activation
- **Mish** - Self-regularized activation
- **GLU** - Gated Linear Unit
- **Hardswish** - Efficient Swish approximation

### Smooth Activations
- **Softplus** - Smooth ReLU approximation (always positive)
- **Softsign** - Polynomial alternative to tanh

### Efficient Mobile Activations
- **Hardsigmoid** - Linear sigmoid approximation
- **Hardtanh** - Linear tanh approximation

### Specialized Functions
- **Hardshrink** - Hard thresholding for sparsity
- **Softshrink** - Soft thresholding for sparsity
- **Tanhshrink** - Tanh shrinkage function
- **Threshold** - Configurable step function
- **Softmax2d** - Spatial softmax for segmentation
- **Softmin** - Inverse softmax (emphasizes minimum)

### Advanced NLP Functions
- **AdaptiveLogSoftmax** - Efficient large vocabulary softmax
- **MultiheadAttention** - Core transformer attention mechanism

**Total: 31 activation functions with both PyTorch and TensorFlow implementations**

## Quick Start

1. Clone this repository
2. Open `index.html` in your browser
3. Search and explore activation functions
4. Click on any card to see detailed information

## What Each Modal Provides

- **Mathematical Formula** - The exact mathematical definition
- **Framework Toggle** - Switch between PyTorch and TensorFlow implementations
- **PyTorch Implementation** - Complete code examples with multiple usage patterns
- **TensorFlow Implementation** - Complete TensorFlow/Keras equivalents 
- **Practitioner's Checklist** - When to use, advantages, and potential issues
- **Key Properties** - Output range, smoothness, monotonicity, and centering

## Technology Stack

- **HTML5** - Semantic structure
- **CSS3** - Neumorphic design with custom properties
- **Vanilla JavaScript** - No dependencies, lightweight and fast
- **Font Awesome** - Icons
- **Google Fonts** - Inter typeface

## Implementation Complete 🎉

All activation functions from PyTorch's `torch.nn` module have been implemented with dual framework support:
- ✅ Complete mathematical formulas
- ✅ **PyTorch code examples** - Complete torch.nn and functional implementations
- ✅ **TensorFlow code examples** - Complete tf.keras and tf.nn implementations  
- ✅ **Framework switching** - Toggle between PyTorch and TensorFlow in each modal
- ✅ Practitioner guidance and checklists
- ✅ Interactive search and filtering
- ✅ Modern neumorphic design

The application now covers **all 31 activation functions** with implementations for both major deep learning frameworks.

## Contributing

Feel free to contribute by:
- Adding more activation functions
- Improving the UI/UX
- Adding mathematical visualizations
- Enhancing the practitioner checklists
- Adding performance benchmarks

## License

MIT License - Feel free to use this for educational and commercial purposes.