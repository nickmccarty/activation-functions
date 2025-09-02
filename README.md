# PyTorch & TensorFlow Activation Functions Explorer

[![Demo](https://img.shields.io/badge/Demo-Live%20Site-blue?style=for-the-badge&logo=github)](https://nickmccarty.me/activation-functions)
[![Wizard](https://img.shields.io/badge/Wizard-Interactive%20Guide-purple?style=for-the-badge&logo=magic)](https://nickmccarty.me/activation-functions/activation_wizard.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-Ready-ee4c2c?style=flat&logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Ready-ff6f00?style=flat&logo=tensorflow)](https://tensorflow.org/)

An interactive, modern web application for exploring activation functions across **both PyTorch and TensorFlow** with detailed explanations, code examples, practitioner guidance, and an **intelligent wizard** to help you choose the perfect activation function for your use case.

## Features

### Core Explorer
- 🔍 **Smart Search** - Find activation functions by name, description, or properties
- 🎨 **Neumorphic Design** - Modern, tactile UI with smooth animations
- 📱 **Responsive** - Works seamlessly on desktop and mobile devices
- 💡 **Interactive Cards** - Click any function for detailed information
- 📊 **Practitioner Checklists** - Actionable guidance for choosing the right activation
- 🔄 **Dual Framework Support** - Switch between PyTorch and TensorFlow implementations
- 🐍 **PyTorch Code** - Complete, runnable PyTorch examples
- 🔶 **TensorFlow Code** - Complete, runnable TensorFlow/Keras examples
- 🏷️ **Smart Filtering** - Category-based filtering (Common, Advanced, Smooth)

### 🧙‍♂️ Interactive Wizard
- 🎯 **Personalized Recommendations** - Get activation function suggestions based on your specific use case
- 📋 **Step-by-Step Guidance** - Wizard walks you through architecture decisions
- 📊 **Performance Visualizations** - Compare computational costs with interactive charts
- 💻 **Implementation Examples** - See complete code examples for your chosen functions
- 🎨 **Seamless Integration** - Launches in modal overlay without leaving the main app

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
3. **Use the Wizard**: Click "Find My Activation" to get personalized recommendations
4. **Browse & Explore**: Search and filter through all 31 activation functions
5. **Deep Dive**: Click on any card to see detailed implementation information

### Two Ways to Explore

#### 🧙‍♂️ **Guided Experience** (Recommended for beginners)
Click the **"Find My Activation"** button to launch the interactive wizard that will:
- Ask about your use case (output layer, hidden layers, special requirements)
- Guide you through architecture decisions
- Provide personalized recommendations with pros/cons analysis
- Show implementation code for your specific needs

#### 🔍 **Free Exploration** (Great for browsing)
Use the main interface to:
- Search through all activation functions
- Filter by category (Common, Advanced, Smooth)
- Click cards for detailed technical information
- Compare PyTorch and TensorFlow implementations

## What Each Modal Provides

- **Mathematical Formula** - The exact mathematical definition
- **Framework Toggle** - Switch between PyTorch and TensorFlow implementations
- **PyTorch Implementation** - Complete code examples with multiple usage patterns
- **TensorFlow Implementation** - Complete TensorFlow/Keras equivalents 
- **Practitioner's Checklist** - When to use, advantages, and potential issues
- **Key Properties** - Output range, smoothness, monotonicity, and centering

## Technology Stack

- **HTML5** - Semantic structure with modal overlays
- **CSS3** - Neumorphic design with custom properties and responsive layouts
- **Vanilla JavaScript** - No dependencies, lightweight and fast
- **Plotly.js** - Interactive performance visualizations in wizard
- **Font Awesome** - Icons and visual elements
- **Google Fonts** - Inter typeface for modern typography

## Implementation Complete 🎉

All activation functions from PyTorch's `torch.nn` module have been implemented with dual framework support:
- ✅ Complete mathematical formulas
- ✅ **PyTorch code examples** - Complete torch.nn and functional implementations
- ✅ **TensorFlow code examples** - Complete tf.keras and tf.nn implementations  
- ✅ **Framework switching** - Toggle between PyTorch and TensorFlow in each modal
- ✅ **Interactive wizard** - Personalized recommendations based on use case
- ✅ **Performance visualizations** - Computational cost comparisons
- ✅ Practitioner guidance and checklists
- ✅ Interactive search and filtering
- ✅ Modern neumorphic design with responsive layouts
- ✅ **Jupyter notebook guide** - Executable examples for hands-on learning

The application now covers **all 31 activation functions** with implementations for both major deep learning frameworks, plus intelligent guidance tools.

## Contributing

Feel free to contribute by:
- Adding more activation functions
- Improving the UI/UX
- Adding mathematical visualizations
- Enhancing the practitioner checklists
- Adding performance benchmarks

## License

MIT License - Feel free to use this for educational and commercial purposes.