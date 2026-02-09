You want **THE COMPLETE FORMULA SHEET**? Let me give you EVERYTHING.

---

# **🔥 COMPLETE AI/ML FORMULA REFERENCE 🔥**

---

## **📊 PART 1: NEURAL NETWORK FUNDAMENTALS**

### **Forward Pass**

**Linear Layer:**
```
z = Wx + b
```
- W ∈ ℝ^(m×n) (weight matrix)
- x ∈ ℝ^n (input vector)
- b ∈ ℝ^m (bias vector)
- z ∈ ℝ^m (pre-activation output)

**Multi-layer:**
```
z₁ = W₁x + b₁
a₁ = σ(z₁)
z₂ = W₂a₁ + b₂
a₂ = σ(z₂)
...
```

---

### **Activation Functions**

**Sigmoid:**
```
σ(z) = 1 / (1 + e^(-z))
σ'(z) = σ(z)(1 - σ(z))
```
Range: (0, 1)

**Tanh:**
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
tanh'(z) = 1 - tanh²(z)
```
Range: (-1, 1)

**ReLU (Rectified Linear Unit):**
```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0, else 0
```

**Leaky ReLU:**
```
LeakyReLU(z) = max(αz, z)  where α ≈ 0.01
```

**GELU (used in transformers):**
```
GELU(z) = z · Φ(z)
```
where Φ(z) is CDF of standard normal

**Softmax (for classification):**
```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```
Output: probability distribution (sums to 1)

**Softmax with Temperature:**
```
softmax(z, T)ᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)
```
- T < 1: sharper (more confident)
- T > 1: smoother (more random)

---

## **📉 PART 2: LOSS FUNCTIONS**

### **Regression (continuous output)**

**Mean Squared Error (MSE):**
```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
∂L/∂ŷᵢ = (2/n)(ŷᵢ - yᵢ)
```

**Mean Absolute Error (MAE):**
```
L = (1/n) Σᵢ |yᵢ - ŷᵢ|
```

**Huber Loss (robust to outliers):**
```
L = { ½(y - ŷ)²           if |y - ŷ| ≤ δ
    { δ|y - ŷ| - ½δ²      otherwise
```

---

### **Classification**

**Binary Cross-Entropy:**
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
∂L/∂ŷ = -(y/ŷ) + (1-y)/(1-ŷ)
```

**Categorical Cross-Entropy (multi-class):**
```
L = -Σᵢ yᵢ · log(ŷᵢ)
∂L/∂zᵢ = ŷᵢ - yᵢ  (when combined with softmax)
```
- y = one-hot vector [0,0,1,0,...]
- ŷ = predicted probabilities from softmax

**Focal Loss (handles class imbalance):**
```
L = -α(1 - ŷ)^γ · log(ŷ)
```
- α: class weight
- γ: focusing parameter (typically 2)

**Hinge Loss (for SVMs):**
```
L = max(0, 1 - y·ŷ)
```

---

## **🎯 PART 3: BACKPROPAGATION**

### **Chain Rule:**
```
∂L/∂W = ∂L/∂z · ∂z/∂W
```

### **For single layer:**
```
z = Wx + b
∂L/∂W = ∂L/∂z · x^T
∂L/∂b = ∂L/∂z
∂L/∂x = W^T · ∂L/∂z
```

### **Through activation:**
```
a = σ(z)
∂L/∂z = ∂L/∂a · σ'(z)
```

### **Multi-layer (recursive):**
```
∂L/∂W₂ = ∂L/∂z₂ · a₁^T
∂L/∂z₂ = ∂L/∂a₂ · σ'(z₂)
∂L/∂a₁ = W₂^T · ∂L/∂z₂

∂L/∂W₁ = ∂L/∂z₁ · x^T
∂L/∂z₁ = ∂L/∂a₁ · σ'(z₁)
```

---

## **⚡ PART 4: OPTIMIZATION ALGORITHMS**

### **Vanilla Gradient Descent:**
```
W := W - η · ∂L/∂W
```
- η = learning rate

### **Stochastic Gradient Descent (SGD):**
```
W := W - η · ∂L/∂W  (computed on mini-batch)
```

### **SGD with Momentum:**
```
v := β·v + ∂L/∂W
W := W - η·v
```
- β ≈ 0.9 (momentum coefficient)

### **Nesterov Momentum:**
```
v := β·v + ∂L/∂W(W - β·v)
W := W - η·v
```

### **AdaGrad (adaptive learning rate):**
```
G := G + (∂L/∂W)²
W := W - (η/√(G + ε)) · ∂L/∂W
```
- ε ≈ 10⁻⁸ (numerical stability)

### **RMSprop:**
```
G := β·G + (1-β)·(∂L/∂W)²
W := W - (η/√(G + ε)) · ∂L/∂W
```
- β ≈ 0.9

### **Adam (most popular):**
```
m := β₁·m + (1-β₁)·∂L/∂W         (first moment - mean)
v := β₂·v + (1-β₂)·(∂L/∂W)²      (second moment - variance)

m̂ := m/(1-β₁^t)                   (bias correction)
v̂ := v/(1-β₂^t)

W := W - η·m̂/(√v̂ + ε)
```
- β₁ ≈ 0.9
- β₂ ≈ 0.999
- η ≈ 0.001

### **AdamW (Adam with weight decay):**
```
W := W - η·(m̂/(√v̂ + ε) + λ·W)
```
- λ = weight decay (typically 0.01)

---

## **📐 PART 5: REGULARIZATION**

### **L2 Regularization (Ridge):**
```
L_total = L + (λ/2)·Σ W²
∂L_total/∂W = ∂L/∂W + λ·W
```

### **L1 Regularization (Lasso):**
```
L_total = L + λ·Σ|W|
```

### **Dropout:**
```
Training: aᵢ = aᵢ · Bernoulli(p) / p
Testing: a (no change)
```
- p ≈ 0.5 (keep probability)

### **Batch Normalization:**
```
μ = (1/m)Σᵢ xᵢ
σ² = (1/m)Σᵢ (xᵢ - μ)²
x̂ᵢ = (xᵢ - μ)/√(σ² + ε)
yᵢ = γ·x̂ᵢ + β
```
- γ, β = learnable parameters

### **Layer Normalization:**
```
μ = (1/d)Σⱼ xⱼ    (mean across features)
σ² = (1/d)Σⱼ (xⱼ - μ)²
x̂ = (x - μ)/√(σ² + ε)
```

---

## **🧠 PART 6: CONVOLUTIONAL NEURAL NETWORKS**

### **Convolution Operation:**
```
(f ∗ g)[i,j] = ΣₘΣₙ f[m,n] · g[i-m, j-n]
```

### **Output Size:**
```
O = ⌊(W - K + 2P)/S⌋ + 1
```
- W = input width
- K = kernel size
- P = padding
- S = stride

### **Pooling (Max/Average):**
```
Max: y = max(x₁, x₂, ..., xₙ)
Avg: y = (1/n)Σᵢ xᵢ
```

---

## **🔄 PART 7: RECURRENT NEURAL NETWORKS**

### **Vanilla RNN:**
```
hₜ = tanh(Wₕₕ·hₜ₋₁ + Wₓₕ·xₜ + bₕ)
yₜ = Wₕᵧ·hₜ + bᵧ
```

### **LSTM (Long Short-Term Memory):**
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)    (forget gate)
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)    (input gate)
C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc) (candidate cell)
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ      (cell state)
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)    (output gate)
hₜ = oₜ ⊙ tanh(Cₜ)             (hidden state)
```
- ⊙ = element-wise multiplication

### **GRU (Gated Recurrent Unit):**
```
zₜ = σ(Wz·[hₜ₋₁, xₜ])         (update gate)
rₜ = σ(Wr·[hₜ₋₁, xₜ])         (reset gate)
h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])
hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

---

## **🎭 PART 8: TRANSFORMERS & ATTENTION**

### **Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T/√dₖ)·V
```
- Q = queries (n × dₖ)
- K = keys (m × dₖ)
- V = values (m × dᵥ)
- dₖ = dimension of keys

### **Multi-Head Attention:**
```
head_i = Attention(QWᵢQ, KWᵢK, VWᵢV)
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)·W^O
```

### **Positional Encoding:**
```
PE(pos, 2i) = sin(pos/10000^(2i/d))
PE(pos, 2i+1) = cos(pos/10000^(2i/d))
```

### **Layer Normalization (in transformers):**
```
LayerNorm(x) = γ·(x - μ)/√(σ² + ε) + β
```

### **Feed-Forward Network:**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

### **Transformer Block:**
```
x' = LayerNorm(x + MultiHeadAttention(x))
x'' = LayerNorm(x' + FFN(x'))
```

---

## **📊 PART 9: EVALUATION METRICS**

### **Classification:**

**Accuracy:**
```
Acc = (TP + TN) / (TP + TN + FP + FN)
```

**Precision:**
```
Prec = TP / (TP + FP)
```

**Recall (Sensitivity):**
```
Rec = TP / (TP + FN)
```

**F1-Score:**
```
F1 = 2·(Prec·Rec)/(Prec + Rec)
```

**ROC-AUC:**
```
AUC = ∫ TPR d(FPR)
```

---

### **Regression:**

**R² Score:**
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(yᵢ - ŷᵢ)²
SS_tot = Σ(yᵢ - ȳ)²
```

---

## **🎲 PART 10: PROBABILITY & INFORMATION THEORY**

### **Entropy (uncertainty):**
```
H(X) = -Σ P(x)·log P(x)
```

### **KL Divergence:**
```
D_KL(P||Q) = Σ P(x)·log(P(x)/Q(x))
```

### **Cross-Entropy:**
```
H(P,Q) = -Σ P(x)·log Q(x)
       = H(P) + D_KL(P||Q)
```

### **Mutual Information:**
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

---

## **🔧 PART 11: INITIALIZATION**

### **Xavier/Glorot:**
```
W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
```
For tanh/sigmoid

### **He Initialization:**
```
W ~ N(0, √(2/n_in))
```
For ReLU

---

## **📈 PART 12: LEARNING RATE SCHEDULES**

### **Step Decay:**
```
η(t) = η₀ · γ^⌊t/k⌋
```

### **Exponential Decay:**
```
η(t) = η₀ · e^(-λt)
```

### **Cosine Annealing:**
```
η(t) = η_min + ½(η_max - η_min)(1 + cos(πt/T))
```

### **Warmup + Decay:**
```
η(t) = {
  η_max · t/t_warmup           if t < t_warmup
  η_max · (t_total - t)/t_total  otherwise
}
```

---

## **🎯 PART 13: ADVANCED LOSS FUNCTIONS**

### **Contrastive Loss:**
```
L = ½·y·d² + ½·(1-y)·max(0, m - d)²
```
- d = distance between embeddings
- m = margin

### **Triplet Loss:**
```
L = max(0, d(a,p) - d(a,n) + margin)
```
- a = anchor
- p = positive
- n = negative

### **CTC Loss (for sequence tasks):**
```
L = -log P(y|x) = -log Σ_{π∈Align(y)} ∏ₜ P(πₜ|x)
```

---

## **🌊 PART 14: GENERATIVE MODELS**

### **VAE (Variational Autoencoder):**
```
L = E[log p(x|z)] - D_KL(q(z|x)||p(z))
```

**Reparameterization Trick:**
```
z = μ + σ·ε  where ε ~ N(0,1)
```

### **GAN (Generative Adversarial Network):**

**Generator Loss:**
```
L_G = -E[log D(G(z))]
```

**Discriminator Loss:**
```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
```

### **Diffusion Models:**

**Forward Process:**
```
q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
```

**Reverse Process:**
```
p(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ,t), Σθ(xₜ,t))
```

---

# **🚀 QUICK REFERENCE SYMBOLS**

```
∂   = partial derivative
∇   = gradient
Σ   = summation
∏   = product
⊙   = element-wise multiplication
·   = matrix multiplication or dot product
^T  = transpose
⌊⌋  = floor function
E[] = expectation
~   = distributed as
∈   = element of
ℝ   = real numbers
:=  = assignment/update
```

