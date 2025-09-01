# PyTorch Activation Functions Explorer

[![Demo](https://img.shields.io/badge/Demo-Live%20Site-blue?style=for-the-badge&logo=github)](https://nickmccarty.me/activation-functions)

An interactive, modern web application for exploring PyTorch activation functions with detailed explanations, code examples, and practitioner guidance.

## Features

- 🔍 **Smart Search** - Find activation functions by name, description, or properties
- 🎨 **Neumorphic Design** - Modern, tactile UI with smooth animations
- 📱 **Responsive** - Works seamlessly on desktop and mobile devices
- 💡 **Interactive Cards** - Click any function for detailed information
- 📊 **Practitioner Checklists** - Actionable guidance for choosing the right activation
- 🐍 **PyTorch Code** - Complete, runnable examples for each function
- 🏷️ **Smart Filtering** - Category-based filtering (Common, Advanced, Smooth)

## Currently Included Functions

### Common Activations
- **ReLU** - Rectified Linear Unit
- **Sigmoid** - Classic S-shaped activation
- **Tanh** - Hyperbolic Tangent
- **Softmax** - Probability distribution function

### Advanced Activations
- **GELU** - Gaussian Error Linear Unit (Transformer favorite)
- **ELU** - Exponential Linear Unit
- **SELU** - Scaled ELU (Self-normalizing)
- **LeakyReLU** - ReLU with small negative slope
- **PReLU** - Parametric ReLU with learnable slope
- **SiLU/Swish** - Smooth self-gating activation
- **Mish** - Self-regularized activation
- **GLU** - Gated Linear Unit
- **Hardswish** - Efficient Swish approximation

## Quick Start

1. Clone this repository
2. Open `index.html` in your browser
3. Search and explore activation functions
4. Click on any card to see detailed information

## What Each Modal Provides

- **Mathematical Formula** - The exact mathematical definition
- **PyTorch Implementation** - Complete code examples with multiple usage patterns
- **Practitioner's Checklist** - When to use, advantages, and potential issues
- **Key Properties** - Output range, smoothness, monotonicity, and centering

## Technology Stack

- **HTML5** - Semantic structure
- **CSS3** - Neumorphic design with custom properties
- **Vanilla JavaScript** - No dependencies, lightweight and fast
- **Font Awesome** - Icons
- **Google Fonts** - Inter typeface

## TODO: Functions to Add Later

### Non-linear Activations (Weighted Sum)
- [ ] **Hardshrink** - Hard shrinkage function
- [ ] **Hardsigmoid** - Efficient sigmoid approximation  
- [ ] **Hardtanh** - Efficient tanh approximation
- [ ] **LogSigmoid** - Logarithm of sigmoid
- [ ] **ReLU6** - ReLU clamped at 6
- [ ] **RReLU** - Randomized Leaky ReLU
- [ ] **CELU** - Continuously differentiable ELU
- [ ] **Softplus** - Smooth approximation of ReLU
- [ ] **Softshrink** - Soft shrinkage function
- [ ] **Softsign** - Softsign activation
- [ ] **Tanhshrink** - Tanh shrinkage function
- [ ] **Threshold** - Threshold activation

### Non-linear Activations (Other)
- [ ] **Softmin** - Softmin function
- [ ] **Softmax2d** - Spatial softmax
- [ ] **LogSoftmax** - Log of softmax
- [ ] **AdaptiveLogSoftmaxWithLoss** - Efficient softmax approximation

### Attention Mechanisms
- [ ] **MultiheadAttention** - Multi-head attention mechanism

## Contributing

Feel free to contribute by:
- Adding more activation functions
- Improving the UI/UX
- Adding mathematical visualizations
- Enhancing the practitioner checklists
- Adding performance benchmarks

## License

MIT License - Feel free to use this for educational and commercial purposes.