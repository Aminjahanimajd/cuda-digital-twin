# CUDA p-Bit Route Planner for Digital Asimov

A compact educational demonstration of probabilistic Ising optimization on an
NVIDIA GPU, followed by execution of the selected route in Menlo's Digital
Asimov humanoid simulator.

The project implements this complete pipeline in one Python program:

```text
Fixed route candidates
        |
        v
Direct Ising formulation (h, J)
        |
        v
CUDA p-bit updates over 4,096 replicas
        |
        v
GPU population annealing and decoding
        |
        v
Selected action sequence: BLFL
        |
        v
Digital Asimov PPO locomotion
```

The main objective is to demonstrate a direct translation from a GPU
Monte-Carlo-style probabilistic spin solver to GPU p-bits. The example is kept
intentionally small so that the complete optimization and robot-control flow
can be understood from a single source file.

## Key features

- Direct Ising construction: no QUBO matrix and no QUBO-to-Ising conversion.
- A custom CUDA C p-bit update implemented with `cupy.RawKernel`.
- `4,096` probabilistic replicas processed in parallel on the GPU.
- Population annealing over `40` inverse-temperature values.
- Boltzmann-weighted GPU resampling between temperature steps.
- Reproducible stochastic execution using the fixed seed `7`.
- Exactly-one-route constraint represented through Ising couplings.
- Digital Asimov control through the Menlo Python SDK.
- Execution through Menlo's existing PPO-trained locomotion policy.
- Cleanup of the temporary simulator robot even if execution fails.

## Division of responsibility

The project has two separate computational parts:

| Component | Responsibility |
|---|---|
| CUDA p-bit solver | Selects one low-cost valid route. |
| Digital Asimov PPO policy | Converts velocity commands into balanced humanoid walking. |

PPO stands for **Proximal Policy Optimization**. This project does not train a
new reinforcement-learning policy. It uses Menlo's already-trained locomotion
controller, while the route-selection computation is performed by the CUDA
p-bit solver.

## Route-planning problem

Four fixed candidates are stored in `ROUTES`. A route contains a descriptive
name, a numerical cost, and an ordered action string.

| Spin index | Route | Cost | Actions | Interpretation |
|---:|---|---:|---|---|
| 0 | Direct, blocked by table | 22.0 | `F` | Forward motion is discouraged by a large obstacle cost. |
| 1 | Safe left detour | 5.0 | `BLFL` | Lowest-cost valid route and intended solution. |
| 2 | Safe right detour | 5.4 | `BRFR` | Valid but slightly more expensive alternative. |
| 3 | Conservative detour | 7.0 | `BBLF` | Longer route with additional clearance. |

The action letters have fixed meanings:

| Action | Velocity command | Meaning |
|---|---|---|
| `B` | `vx=-0.8 m/s` | Move backward. |
| `F` | `vx=+0.8 m/s` | Move forward. |
| `L` | `vy=+0.8 m/s` | Strafe left. |
| `R` | `vy=-0.8 m/s` | Strafe right. |

Each motion lasts `1.0 s`. The program leaves a `1.5 s` recovery interval
between successive commands.

These costs are part of the demonstration model. In a larger application they
could be calculated from distance, time, energy use, terrain, or collision
risk instead of being written manually.

## Direct Ising formulation

Each route is represented by one Ising spin:

```text
s_i = +1  -> route i is selected
s_i = -1  -> route i is not selected
```

The binary interpretation is:

```text
x_i = (s_i + 1) / 2
```

The intended optimization objective is:

```text
C(x) = sum(c_i * x_i) + A * (sum(x_i) - 1)^2
```

where:

- `c_i` is the cost of route `i`;
- `A` is `PENALTY = 30.0`;
- the first term prefers inexpensive routes;
- the second term penalizes selecting zero routes or more than one route.

The program writes this objective directly in Ising form:

```text
E(s) = sum(h_i * s_i) + sum(J_ij * s_i * s_j)
```

For `n` routes, the code constructs:

```python
h = costs / 2 + PENALTY * (n - 2) / 2
J = full((n, n), PENALTY / 2)
diagonal(J) = 0
```

`h` contains the individual route preference and part of the exactly-one
penalty. `J` couples every pair of route spins, making invalid multi-route
selections expensive. This construction is direct; the program never creates
a QUBO object.

## CUDA p-bit kernel

The custom CUDA kernel receives the spin population `s`, Ising parameters `h`
and `J`, random values `rnd`, the current spin index `bit`, the number of spins
`n`, the population size `replicas`, and inverse temperature `beta`.

For one replica, it calculates the local field of one spin:

```c
float local = h[bit];
for (int j = 0; j < n; j++)
    local += J[bit*n + j] * s[r*n + j];
```

It then samples the updated spin probabilistically:

```c
float probability_plus = 1.0f / (1.0f + expf(2.0f*beta*local));
s[r*n + bit] = rnd[r] < probability_plus ? 1 : -1;
```

`local` describes the preference of one particular spin, while the full
Ising energy evaluates all four spins of a replica together.

Each kernel launch uses blocks of `256` CUDA threads. For one spin position,
one logical thread updates one replica, so all `4,096` replicas are exposed to
GPU parallelism. The four spin positions are updated sequentially because the
new value of one spin should influence later spin updates.

## Population annealing

The population begins as a random `4096 x 4` array of `-1` and `+1` values.
Every row is one replica: a possible route-selection answer.

The inverse-temperature schedule is:

```python
betas = cp.linspace(0.1, 5.0, 40)
```

At low `beta`, updates are relatively random and explore different spin
configurations. As `beta` increases, low-energy states become increasingly
probable.

For every temperature after the first, the code computes each replica's full
energy and its normalized Boltzmann weight:

```python
weights = exp(-(beta - old_beta) * (energy - minimum_energy))
weights /= weights.sum()
```

The population is resampled with replacement using these weights. Good
low-energy replicas are therefore likely to produce more copies, while poor
replicas gradually disappear. Two complete p-bit sweeps are then performed at
the current temperature.

The solver performs:

```text
40 beta steps x 2 sweeps x 4 spins x 4,096 replicas
= 1,310,720 logical replica-spin updates
```

The work is small for a modern GPU, but it clearly demonstrates the same
parallel structure that can be extended to larger Ising problems.

## Decoding the result

After the final temperature step, the program converts spins to binary route
selections and keeps only replicas satisfying:

```python
selected.sum(axis=1) == 1
```

Among those valid replicas, it chooses the one with the smallest original
route cost. With the fixed seed and current costs, the expected decoded result
is:

```text
Final spins: [-1, 1, -1, -1]
Selected route: safe left detour
Selected route actions: BLFL
```

The value printed by the current program as `Final energy: 5.0` is the decoded
route cost returned by `solve_route()`. It is not the unshifted numerical value
of the complete Ising Hamiltonian.

## Forward-movement safety rule

The selected route uses this sequence:

```text
B -> L -> F -> L
```

The variable `forward_credit` implements a simple rule:

1. Every `B` adds one credit.
2. Every `F` requires and consumes one credit.
3. `L` and `R` do not change the credit.

For `BLFL`, the credit sequence is:

```text
B: 1 credit
L: 1 credit
F: 0 credits
L: 0 credits
```

Consequently, forward movement cannot occur unless an earlier backward command
has created clearance from the table. This is a small open-loop safety rule,
not perception-based collision avoidance.

## Digital Asimov execution

After decoding, `run_robot()` performs the following steps:

1. Creates a temporary `asimov-v0` robot.
2. Connects a Menlo SDK session using the browser simulator as the runtime.
3. Generates and prints a temporary simulator URL.
4. Waits until the browser joins and exposes its skills.
5. Waits `15 s` for the locomotion policy to stabilize.
6. Sends each decoded velocity command in order.
7. Sends zero velocity when the route finishes.
8. Disconnects and deletes the temporary robot in a `finally` block.

The simulator invocation timeout is `15 s`. If an action may have been sent but
its acknowledgement is lost, `lost_replies` records the communication problem
and the program continues without blindly repeating the motion. This avoids
duplicating a command whose execution status is uncertain.

Keep the simulator tab open and active during execution. That browser tab is
the Digital Asimov runtime and must remain connected to answer SDK calls.

## Project structure

```text
.
|-- main.py                                # Complete solver and robot workflow
|-- requirements.txt                       # Python dependencies
|-- README.md                              # Project documentation
|-- CUDA_pBit_Digital_Asimov_Report.docx   # Companion academic report
|-- digital_asimov_left_detour.mp4         # Recorded simulator evidence
`-- robot_route_frames/                    # Before/after evidence frames
```

`main.py` is the only executable project source file. CUDA C is embedded inside
it through `cupy.RawKernel`.

## Requirements

- Windows or Linux with Python `3.10` or newer.
- An NVIDIA CUDA-capable GPU and compatible NVIDIA driver.
- Internet access and a modern Chromium-based browser.
- A Menlo Platform account and API key.
- The packages listed in `requirements.txt`:

```text
cupy-cuda13x[ctk]>=14,<15
menlo-robot-sdk[livekit]==0.3.0
```

The CuPy `[ctk]` extra installs NVIDIA CUDA component wheels into the Python
environment. A separate system-wide `nvcc` installation is therefore not
required for this demonstration, although a compatible NVIDIA driver is still
required.

## Installation on Windows PowerShell

Clone the repository and enter it:

```powershell
git clone https://github.com/Aminjahanimajd/cuda-digital-twin.git
cd cuda-digital-twin
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Menlo API key

Sign in at [platform.menlo.ai](https://platform.menlo.ai/), then open
**Settings -> API Keys** and create a key. Set it only in the current terminal:

```powershell
$env:MENLO_API_KEY="your-key-here"
$env:MENLO_RCS_URL="https://api.menlo.ai/rcs"
```

`MENLO_RCS_URL` is optional because `main.py` already uses the displayed URL as
its default.

Never paste a real API key into `main.py`, `README.md`, screenshots, commits, or
GitHub issues. `.menlo-key` is excluded by `.gitignore` as an additional
precaution.

## Running the demonstration

Start the complete workflow:

```powershell
python main.py
```

The terminal should first show output similar to:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Final spins: [-1, 1, -1, -1]
Final energy: 5.0
Selected route: safe left detour
```

The program then prints a temporary `https://sim.menlo.ai/?key=...` URL.

1. Copy the complete URL.
2. Open it in Chrome.
3. Keep the simulator tab open and visible.
4. Wait for `Simulator connected.` in PowerShell.
5. Observe the `BLFL` movement and terminal acknowledgements.

A fully acknowledged execution ends with:

```text
Executing table-avoidance actions: BLFL
B done forward credit: 1
L done forward credit: 1
F done forward credit: 0
L done forward credit: 0
Task complete. Robot executed the safe detour and stopped.
```

If the browser executes a command but loses its RPC acknowledgement, the
terminal may instead report that its reply was lost. This is a communication
warning, not evidence of an obstacle collision.

## Troubleshooting

### `Set MENLO_API_KEY before running this program`

Set the environment variable in the same PowerShell window where you run
Python:

```powershell
$env:MENLO_API_KEY="your-key-here"
python main.py
```

### CuPy cannot detect the GPU

Confirm that Windows detects the NVIDIA GPU and that the driver is current:

```powershell
nvidia-smi
```

Also confirm that only one CuPy distribution is installed:

```powershell
python -m pip list | Select-String cupy
```

### Program remains at `Waiting for the browser simulator...`

Open the complete simulator URL printed by the current run. A URL from an older
run belongs to a different temporary robot and should not be reused.

### `Connection timeout` or a lost reply

Keep Chrome open and focused, maintain a stable network connection, and avoid
refreshing the simulator while commands are running. The code waits up to
`15 s` for each acknowledgement and reports uncertain replies without
automatically repeating the corresponding motion.

## Scope and limitations

- The four routes and their costs are predefined; the robot does not generate
  them from camera images.
- The obstacle is modeled through route cost and the movement-credit rule; it
  is not detected dynamically.
- Execution is open-loop and does not use global position feedback to correct
  accumulated motion error.
- The four-route example is too small to demonstrate a speed advantage over a
  normal `min()` operation or a classical shortest-path algorithm.
- The purpose is to demonstrate the CUDA probabilistic optimization workflow,
  not to claim superior performance for this particular input.
- PPO locomotion is supplied by Menlo; the project does not train or modify the
  reinforcement-learning policy.
- QUBO construction, D-Wave embedding, quantum annealing, inverse kinematics,
  perception, and sim-to-real deployment are outside the implementation.
- A probabilistic solver can produce different populations when its random seed
  changes, even though the present fixed seed makes the demonstration
  reproducible.

## Suggested presentation sequence

1. Introduce the route-selection task and four fixed candidate costs.
2. Explain that one Ising spin represents each route.
3. Show `h`, `J`, `REPLICAS`, and `PENALTY` in `main.py`.
4. Point to the CUDA `RawKernel` and its probabilistic spin update.
5. Explain the `40` beta steps, two sweeps, and population resampling.
6. Run the program and identify the GPU, spins, cost, and decoded route.
7. Open Digital Asimov and show the PPO controller executing `BLFL`.
8. Finish with the honest limitation: the problem is educational and small,
   while the solver structure is intended to illustrate larger parallel cases.

One-sentence summary:

> This project uses CUDA to update and anneal thousands of probabilistic Ising
> replicas, decodes the lowest-cost valid route, and sends its velocity sequence
> to a PPO-controlled humanoid digital twin.

## References

- [Menlo Platform Python SDK quickstart](https://docs.menlo.ai/platform/get-started/quickstart)
- [Digital Asimov documentation](https://docs.menlo.ai/asimov/digital-asimov)
- [Asimov robot control and velocity commands](https://docs.menlo.ai/asimov/1/api/robot-control)
- [Menlo reinforcement-learning simulation environment](https://docs.menlo.ai/guides/locomotion-training/reinforcement-learning-simulation-training-environment)
- [CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html)
- [Quantum annealing for inverse kinematics in robotics](https://www.nature.com/articles/s41598-025-34346-z)
- [How we are leading a 3-XORSAT challenge](https://iopscience.iop.org/article/10.1209/0295-5075/133/60005/meta)
- [GPU accelerated population annealing algorithm](https://arxiv.org/abs/1703.03676)

## Academic integrity note

This repository is an educational demonstration. Results should be presented
with the stated assumptions and limitations, and external code, papers, and
platform services should be cited appropriately.
