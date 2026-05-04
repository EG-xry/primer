# Retina Module Code Reproduction Plan

## Overview

This plan aims to reproduce the complete analysis pipeline of the `RetinaScripts/` module, which systematically analyzes the responses of retinal ganglion cells (RGCs) to white noise checkerboard stimuli, progressively building neural coding models from simple to complex.

**Paper Reference:** Aljadeff, Lansdell, Fairhall & Kleinfeld (2016) — *Analysis of Neuronal Spike Trains, Deconstructed*

---

## Stage 1: Data Preparation and Preprocessing

### Step 1.1 — Build Stimulus Matrix and Response Vector
**Corresponding script:** `script_101_RetinaData_BuildStimulus.m`

**Objective:** Extract and organize stimulus and response data from raw binary files

**Input data:**
- `whitenoise.raw` — Raw checkerboard stimulus (binary format)
- `whitenoisec1.isk ~ whitenoisec53.isk` — Spike trains from 53 cells
- `RetinaCellParameters_*.mat` — Cell parameters (spatial receptive field location, size, temporal delay, etc.)

**Key operations:**
1. Use `ReadFramev2.m` to read the raw stimulus file and extract the spatial receptive field region for each cell
2. Reshape the stimulus into a (T × N_X) matrix, where T is the number of time frames and N_X is the spatial dimension
3. Zero-mean and normalize each pixel dimension
4. Align stimulus and response using `circshift` based on each cell's temporal delay parameter
5. For multi-frame temporal dependence (NT > 1), construct a spatiotemporal joint stimulus matrix
6. Split data into training and test sets
7. Perform Jackknife grouping for subsequent cross-validation

**Output:** Preprocessed stimulus matrices `S_train/S_test` and response vectors `R_train/R_test` as `.mat` files

**Reproduction notes:**
- Understand the binary file reading format in `ReadFramev2.m`
- Pay attention to spatial cropping of stimuli and temporal delay alignment
- Default analysis target is cell 3

---

## Stage 2: Linear Model — STA

### Step 2.1 — Compute Spike-Triggered Average (STA)
**Corresponding script:** `script_102_RetinaData_STA.m`

**Objective:** Estimate the neuron's linear receptive field

**Method:**
$$\text{STA} = \frac{1}{n_{\text{spikes}}} \sum_{t: \text{spike at } t} \mathbf{s}(t)$$

**Key operations:**
1. Compute STA for training data in each Jackknife fold
2. STA is the average of stimulus vectors at spike-triggered time points
3. For white noise stimuli, STA is an unbiased estimate of the optimal linear filter

**Output:** STA vector for each Jackknife fold

**Reproduction notes:**
- STA computation is straightforward, no optimization needed
- Note that stimuli are pre-whitened (zero mean, unit variance), so STA ∝ stimulus-response cross-correlation

---

## Stage 3: Second-Order Model — STC

### Step 3.1 — STC Significance Test
**Corresponding script:** `script_103_RetinaData_STC_significance.m`

**Objective:** Compute the spike-triggered covariance matrix and determine significant feature directions

**Method:**
$$C_{\text{STC}} = \frac{1}{n_{\text{spikes}}} \sum_{t: \text{spike at } t} (\mathbf{s}(t) - \text{STA})(\mathbf{s}(t) - \text{STA})^T - C_{\text{prior}}$$

**Key operations:**
1. Compute spike-triggered covariance matrix (STC)
2. Perform eigenvalue decomposition of STC
3. Compare eigenvalues with the Marchenko-Pastur random matrix distribution
4. Determine statistically significant feature directions (eigenvalues exceeding random matrix theory predictions)

**Output:** STC eigenvalues, eigenvectors, significance assessment

### Step 3.2 — Build STC Model
**Corresponding script:** `script_104_RetinaData_STC_model.m`

**Objective:** Construct an LN model using significant STC feature directions

**Key operations:**
1. Select feature directions that pass the significance test
2. Project stimuli onto these directions
3. Estimate the nonlinear function using Bayes' theorem: $P(\text{spike}|\text{projection})$
4. Validate model prediction performance on the test set

**Output:** STC model parameters (significant feature directions + nonlinear function)

**Reproduction notes:**
- Marchenko-Pastur distribution serves as the null hypothesis
- Positive/negative significant eigenvalues correspond to excitatory/inhibitory subspaces respectively

---

## Stage 4: Maximum Noise Entropy Model — MNE

### Step 4.1 — MNE Model Fitting
**Corresponding script:** `script_105_RetinaData_MNE_fitting.m`

**Objective:** Fit a parameterized second-order LN model

**Method (Paper Equation 32):**
$$P(\text{spike}|\mathbf{s}) = \frac{1}{1 + \exp\left(-(A + \mathbf{H}^T\mathbf{s} + \mathbf{s}^T\mathbf{J}\mathbf{s})\right)}$$

Where:
- $A$ — Scalar bias
- $\mathbf{H}$ — Linear feature vector
- $\mathbf{J}$ — Quadratic feature matrix (symmetric)

**Key operations:**
1. Compute constraints: spike probability $p_{sp}$, first-order stimulus-response correlation, second-order correlation
2. Initialize parameters: $A$ from log-odds, $\mathbf{H}$ and $\mathbf{J}$ randomly initialized
3. Minimize log-loss using conjugate gradient method (`frprmn_global_min`)
4. Apply early stopping on the test set to prevent overfitting

**Helper functions:**
- `MNEfit_RetinaData.m` — Main fitting function
- `logloss.m` — Compute log-loss objective function
- `dlogloss.m` — Compute gradient
- `frprmn_global_min.m` — Conjugate gradient optimizer (with early stopping)

**Output:** MNE model parameters for each Jackknife fold

### Step 4.2 — MNE Model Post-processing
**Corresponding script:** `script_106_RetinaData_MNE_model.m`

**Objective:** Extract significant quadratic feature directions

**Key operations:**
1. Average fitting results across Jackknife folds
2. Extract model parameters $A$, $\mathbf{H}$, $\mathbf{J}$
3. Perform eigenvalue decomposition of $\mathbf{J}$ (Equations 34-35)
4. Construct null distribution through permutation test (500 permutations of diagonal/off-diagonal elements of $\mathbf{J}$)
5. Determine statistically significant eigenvalues/feature directions

**Output:** Significant MNE feature directions, eigenvalues, null distribution

**Reproduction notes:**
- MNE is a parameterized alternative to STC that can simultaneously estimate linear and quadratic features
- Early stopping in conjugate gradient is critical for preventing overfitting
- Permutation test is used to determine significance thresholds

---

## Stage 5: Generalized Linear Model — GLM

### Step 5.1 — GLM Fitting
**Corresponding script:** `script_107_RetinaData_GLM.m`

**Objective:** Fit a GLM with spike history

**Model structure:**
$$\lambda(t) = \exp\left(\mathbf{k}^T \mathbf{s}(t) + \mathbf{h}^T \mathbf{r}_{\text{hist}}(t) + b\right)$$

Where:
- $\mathbf{k}$ — Stimulus filter (spatiotemporal bilinear parameterization)
- $\mathbf{h}$ — Post-spike history filter
- $b$ — Baseline firing rate
- $\mathbf{r}_{\text{hist}}$ — Historical spike vector

**Key operations:**
1. Initialize stimulus filter using SVD decomposition of STA (spatial × temporal separation)
2. Construct temporal basis functions (bumps/raised cosines) to parameterize filters
3. Construct post-spike history filter basis functions (with absolute refractory period)
4. Perform maximum likelihood fitting using Jonathan Pillow's GLM toolkit

**Helper functions:**
- `makeFittingStruct_GLM_Retina.m` — Initialize GLM parameter structure

**Dependency:** Jonathan Pillow's GLM toolkit

**Output:** Fitted GLM parameters (stimulus filter + history filter + baseline rate)

**Reproduction notes:**
- GLM adds spike history dependence compared to LN models
- Bilinear (spatial × temporal separation) parameterization significantly reduces parameter count
- Requires installing Pillow's GLM toolkit

---

## Stage 6: Model Prediction and Validation

### Step 6.1 — Generate Predictions
**Corresponding script:** `script_108_RetinaData_Predictions.m`

**Objective:** Generate predicted firing rates on the test set using each model

**Key operations:**
1. For each model (STA-LN, STC-LN, MNE, GLM), compute predicted firing rate from test stimuli through the model
2. STA/STC models: projection + nonlinear mapping
3. MNE model: use fitted logistic function
4. GLM model: use simulation method to generate conditional intensity function

**Output:** Predicted firing rate time series for each model

### Step 6.2 — Validation and Comparison
**Corresponding script:** `script_109_RetinaData_Validation.m`

**Objective:** Quantitatively compare prediction accuracy across models

**Evaluation metrics:**
- Pearson correlation: between predicted firing rate and actual PSTH
- Log-likelihood: goodness of fit of the model to test data
- Information (bits/spike): information ratio captured by the model

**Output:** Performance metric comparison across models

---

## Stage 7: Generate Figures

### Step 7.1 — Visualize Results
**Corresponding scripts:** `makefig_101` ~ `makefig_106`

| Script | Paper Figure | Content |
|--------|-------------|---------|
| `makefig_101` | Figure 3 | Stimulus visualization, PCA eigenspectrum vs. Marchenko-Pastur distribution, principal components |
| `makefig_102` | Figure 4 | STA spatial-temporal filter, projection distributions, nonlinear function, predicted PSTH |
| `makefig_103` | Figure 5 | STC eigenvalue spectrum, significant feature directions, nonlinear functions |
| `makefig_104` | Figure 6 | MNE feature directions, eigenvalues, comparison with STA/STC results |
| `makefig_105` | Figure 7 | GLM stimulus filter, history filter, basis function display |
| `makefig_106` | Figure 8 | Model prediction performance comparison, predicted vs. actual PSTH |

---

## Execution Order Summary

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Data Preparation                               │
│  script_101 → Stimulus matrix + Response vector           │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
│ Stage 2: STA │ │ Stage 3: STC │ │ Stage 4: MNE         │
│ script_102   │ │ script_103   │ │ script_105           │
│              │ │ script_104   │ │ script_106           │
└──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘
       │                │                     │
       │                ▼                     │
       │         ┌──────────────┐             │
       │         │ Stage 5: GLM │             │
       ├────────▶│ script_107   │             │
       │         └──────┬───────┘             │
       │                │                     │
       ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 6: Prediction & Validation                        │
│  script_108 → script_109                                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 7: Generate Figures                               │
│  makefig_101 ~ makefig_106                               │
└─────────────────────────────────────────────────────────┘
```

---

## Environment Configuration Checklist

| Item | Description |
|------|-------------|
| MATLAB Version | R2016a or higher (R2020a+ recommended) |
| Toolkit Dependency | Jonathan Pillow's GLM toolkit |
| Data Files | `whitenoise.raw` + 53 `.isk` files + 3 parameter `.mat` files |
| Memory Requirements | Stimulus matrix is large, ≥ 8GB RAM recommended |
| Computation Time | MNE fitting is most time-consuming (conjugate gradient iterations), other steps are fast |

---

## Reproduction Checkpoints

After completing each stage, verify correctness through the following:

- [ ] **Stage 1**: Check that stimulus matrix dimensions are correct (T × N_X), response vector is binary 0/1
- [ ] **Stage 2**: STA should exhibit a center-surround antagonistic spatial structure
- [ ] **Stage 3**: STC eigenvalue spectrum should have a few significant values exceeding the Marchenko-Pastur boundary
- [ ] **Stage 4**: MNE linear features should be similar to STA, quadratic features should align with STC directions
- [ ] **Stage 5**: GLM stimulus filter should be similar in shape to STA
- [ ] **Stage 6**: GLM ≥ MNE ≥ STC ≥ STA (prediction performance should increase)
- [ ] **Stage 7**: Generated figures should match Figures 3-8 in the paper

---

## Notes

1. **Default analysis cell:** Cell 3, can analyze other cells by modifying the `icell` variable
2. **Stimulus configuration:** Three spatial-temporal dimension configurations (`short2`, `short3`, `long`), default is `short3`
3. **Jackknife folds:** Default 5 folds, used to estimate parameter standard errors
4. **Strict execution order:** Must execute in stage order, subsequent scripts depend on outputs from preceding scripts
5. **GLM toolkit:** Requires separate installation, see [Pillow Lab GLM toolkit](https://github.com/pillowlab/GLMspiketools)
