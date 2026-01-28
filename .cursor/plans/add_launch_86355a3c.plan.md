---
name: ""
overview: ""
todos: []
---

---name: Add Launchoverview: Add Spherical Diffusion Policy (SDP) as a new agent type alongside existing equibot, with separate train/test scripts, reusing per_skill_dataset.py data format and adding text conditioning support.todos:

        - id: add-configs

content: Add 14 launch configurations to .vscode/launch.json with adapted pathsstatus: completed

        - id: todo-1764846602651-yabg1cphg

content: Copy equiformer_v2 and sdp_diffusion modules from code_ref/SDPstatus: completed

        - id: todo-1764846602651-83rber766

content: Create sdp_encoder.py wrapper for EquiformerV2 with text conditioningstatus: completed

        - id: todo-1764846602651-ckzonxwyj

content: Create sdp_policy.py with SDTU and SFiLM conditioningstatus: completed

        - id: todo-1764846602651-qd06rcb36

content: Create sdp_agent.py with train/inference logicstatus: completed

        - id: todo-1764846602651-g90tuwkfd

content: Create sdp_per_skill.yaml config filestatus: completed

        - id: todo-1764846602651-qgjl1v9wp

content: Create train_sdp.py training scriptstatus: completed

        - id: todo-1764846602651-8rltip6hu

content: Create test_sdp.py inference scriptstatus: completed

        - id: todo-1764846602651-mokgtuaap

content: Register sdp agent in misc.py get_agent()status: completed

        - id: todo-1764846602651-som19j326

content: Update setup.py with e3nn, dgl dependenciesstatus: completed

        - id: todo-1764846602651-bp0m9a2g5

content: Add VS Code launch configurations for SDP training/testingstatus: completed

        - id: todo-1764852317060-3fxswz5vg

content: ""status: pending---

# Add SDP Agent for Object-Centric Trajectory Prediction

## Overview

Add Spherical Diffusion Policy (SDP) as an alternative training/inference path, predicting object-centric trajectories conditioned on first-sight point cloud and text inputs (skill/task descriptions). Reuses existing `per_skill_dataset.py` data format.---

## SDP Architecture (from paper)

Reference: `paper_ref/sdp/example_paper.tex` - Section 3 (Method)

### Core Components

1. **EquiformerV2 Encoder** (`enc`)

                                                                - Input: Point cloud observation \(O \in \mathbb{R}^{N \times 6}\) (xyz + rgb)
                                                                - Output: Spherical scene feature \(C\) in Fourier space up to degree \(L\)
                                                                - Achieves SO(3) equivariance via spherical harmonic representations

2. **Spherical Denoising Temporal U-net (SDTU)** (\(\epsilon_\theta\))

                                                                - Estimates noise from noisy actions \(A^k\), step \(k\), conditioned on \(C\)
                                                                - Uses mixing channel temporal convolution (Eq. 4 in paper):

\[h_{l,m,t}^{o} = \sum_{j}\sum_{i \in in} h_{l,m,j}^{i} w_{l,j-t}^{i,o}\]

                                                                - Spatiotemporal equivariant via degree-wise convolution

3. **Spherical FiLM (SFiLM)** Conditioning

                                                                - Equivariant conditioning of denoising on scene features (Eq. 6):

\[\text{SFiLM}(h_l | \gamma_l, \beta_l) = \gamma_l^T h_l \frac{h_l}{||h_l||} + \beta_l\]

                                                                - Supports high-degree Fourier coefficients (vs Vector Neurons' degree-1 only)

4. **Action Representation** (\(\rho_{ee}\))

                                                                - Position: degree-1 vector (3D)
                                                                - Rotation: 3 degree-1 vectors (rotation matrix columns)
                                                                - Gripper: degree-0 scalar
                                                                - Combined: \(\rho_{ee} = \rho_1^4 \oplus \rho_0\)

### Translation Invariance

Achieved via relative action formulation (canonicalization):

```javascript
S_can = (O - e_T, e_T - e_T, e_R, e_grip)
A_can = (A_T - e_T, A_R, A_grip)
```

---

## Implementation Plan

### Step 1: Copy SDP Model Components

Copy from `code_ref/Spherical_Diffusion_Policy/sdp/model/` to `equibot/policies/`:

```javascript
sdp/model/equiformer_v2/     -> equibot/policies/vision/equiformer_v2/
sdp/model/equivariant_vision/ -> equibot/policies/vision/sdp_encoder.py
sdp/model/equivariant_diffusion/ -> equibot/policies/utils/sdp_diffusion/
```

Key files:

- `equiformer_enc.py` - EquiformerV2 encoder
- `irreps_conditional_unet1d.py` - SDTU implementation
- `irreps_conv1d_components.py` - Spherical conv components

### Step 2: Create SDP Policy Class

Create [`equibot/policies/agents/sdp_policy.py`](equibot/policies/agents/sdp_policy.py):

```python
class SDPPolicy(nn.Module):
    """
    Spherical Diffusion Policy for object-centric trajectory prediction.
    
    Inputs:
    - pc: Point cloud [B, 1, N, 3] (first-sight observation)
    - skill_name: Text embedding for skill conditioning
    - task_name: Text embedding for task conditioning
    
    Outputs:
    - eefpos: SE(3) trajectory [B, T, 4, 4]
    - gripper: Gripper actions [B, T, 1]
    """
    def __init__(self, cfg, device="cpu"):
        # 1. EquiformerV2 encoder for point cloud
        self.encoder = EquiFormerEnc(c_dim=cfg.model.c_dim, lmax=cfg.model.lmax, ...)
        
        # 2. Language encoder for text conditioning (reuse from per_skill)
        self.language_encoder = MLPEncoder(...)
        
        # 3. SDTU for denoising (with text conditioning extension)
        self.noise_pred_net = IrrepConditionalUnet1D(
            input_dim=cfg.model.action_irrep_dim,  # rho_1^4 + rho_0
            global_cond_dim=cfg.model.c_dim + cfg.model.text_dim,  # scene + text
            max_lmax=cfg.model.lmax,
        )
```



### Step 3: Create SDP Agent

Create [`equibot/policies/agents/sdp_agent.py`](equibot/policies/agents/sdp_agent.py):

```python
class SDPAgent(DPAgent):
    """Agent wrapper for SDP policy with training/inference logic."""
    
    def update(self, batch):
        # Convert batch to SDP format
        # - pc -> EquiformerV2 input (add robot state if needed)
        # - eefpos -> spherical action representation
        # Train denoising loss
        
    def act(self, obs):
        # Run K-step denoising
        # Convert spherical actions back to SE(3) poses
```



### Step 4: Register New Agent Type

Update [`equibot/policies/utils/misc.py`](equibot/policies/utils/misc.py):

```python
def get_agent(agent_name):
    # ... existing agents ...
    elif agent_name == "sdp":
        from equibot.policies.agents.sdp_agent import SDPAgent
        return SDPAgent
```



### Step 5: Create SDP Config

Create [`equibot/policies/configs/sdp_per_skill.yaml`](equibot/policies/configs/sdp_per_skill.yaml):

```yaml
agent:
  agent_name: sdp

model:
  c_dim: 128
  lmax: 2
  mmax: 2
  
  encoder:
    max_neighbors: [16, 16, 16, 16]
    max_radius: [0.05, 0.2, 0.8, 3]
    pool_ratio: [0.25, 0.25, 0.25]
    sphere_channels: [32, 64, 128]
    
  diffusion:
    down_dims: [200, 400, 800]
    FiLM_type: SFiLM
    num_train_timesteps: 100
```



### Step 6: Create Training Script

Create [`equibot/policies/train_sdp.py`](equibot/policies/train_sdp.py):

```python
@hydra.main(config_path="configs", config_name="sdp_per_skill")
def main(cfg):
    # Reuse PerSkillDataset from per_skill_dataset.py
    dataset = get_dataset(cfg, "train")  # Returns same format
    
    # Initialize SDP agent
    agent = get_agent("sdp")(cfg)
    
    # Training loop (similar to train_skills.py)
    for epoch in range(cfg.training.num_epochs):
        for batch in train_loader:
            # batch contains: pc, skill_name, task_name, eefpos, gripper
            train_metrics = agent.update(batch)
```



### Step 7: Create Inference Script

Create [`equibot/policies/test_sdp.py`](equibot/policies/test_sdp.py):

```python
def run_eval(agent, batch):
    """Evaluate SDP on validation batch."""
    pc = batch['pc']  # First-sight point cloud
    skill_name = batch['skill_name']
    task_name = batch['task_name']
    
    # Predict trajectory
    pred_eefpos, pred_gripper = agent.act({
        'pc': pc,
        'skill_name': skill_name,
        'task_name': task_name
    })
    
    # Compare with ground truth
    gt_eefpos = batch['eefpos']
    metrics = compute_metrics(pred_eefpos, gt_eefpos)
    return metrics
```



### Step 8: Add Dependencies

Update [`setup.py`](setup.py):

```python
install_requires=[
    # ... existing ...
    "e3nn>=0.5.0",        # Spherical harmonics & equivariant ops
    "dgl>=1.0",           # Graph neural network backend
    "pytorch-cluster",     # FPS pooling
    "pytorch-scatter",     # Scatter operations
]
```

---

## Data Flow Diagram

```javascript
per_skill_dataset.py
        |
        v
+------------------+
| pc [B,1,
N
,3]     |  First-sight point cloud
| skill_name       |  Text condition
| task_name        |  Text condition  
| eefpos [B,T,4,4] |  GT trajectory (training only)
| gripper [B,T,1]  |  GT gripper (training only)
+------------------+
        |
        v
+------------------+     +-------------------+
| EquiformerV2     | --> | Scene Feature C   |
| Encoder          |     | (Spherical Fourier)|
+------------------+     +-------------------+
                                 |
        +------------------------+
        |                        |
        v                        v
+------------------+     +-------------------+
| Language Encoder | --> | Text Embedding    |
| (MLPEncoder)     |     | (Invariant)       |
+------------------+     +-------------------+
        |                        |
        +------------------------+
                   |
                   v
        +------------------+
        | SDTU + SFiLM     |  K-step denoising
        | (Conditioned on  |
        |  C + text_emb)   |
        +------------------+
                   |
                   v
        +------------------+
        | eefpos [B,T,4,4] |  Predicted trajectory
        | gripper [B,T,1]  |  Predicted gripper
        +------------------+
```

---

## Key Differences from EquiBot (Vector Neurons)

| Aspect | EquiBot | SDP ||--------|---------|-----|| Equivariance | Vector Neurons (degree-1) | Spherical Harmonics (degree-L) || Feature dim | C x 3 vectors | (L+1)^2 coefficients || Expressiveness | Limited to 3D vectors | Higher-degree spherical signals || Encoder | VecPointNet | EquiformerV2 || Conditioning | VecLNA layers | SFiLM layers |---

## Files to Create/Modify

| Action | Path ||--------|------|| Create | `equibot/policies/vision/equiformer_v2/` (directory) || Create | `equibot/policies/vision/sdp_encoder.py` || Create | `equibot/policies/utils/sdp_diffusion/` (directory) || Create | `equibot/policies/agents/sdp_policy.py` || Create | `equibot/policies/agents/sdp_agent.py` || Create | `equibot/policies/configs/sdp_per_skill.yaml` || Create | `equibot/policies/train_sdp.py` || Create | `equibot/policies/test_sdp.py` || Modify | `equibot/policies/utils/misc.py` || Modify | `setup.py` || Modify | `.vscode/launch.json` (add debug configs) |---

## Testing Strategy

1. **Unit test**: EquiformerV2 encoder SO(3) equivariance
2. **Unit test**: SDTU + SFiLM equivariance  