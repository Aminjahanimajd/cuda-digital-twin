# CUDA p-Bit Planner for Digital Asimov

This is a small classroom demonstration connecting a probabilistic CUDA solver
to the [Digital Asimov](https://docs.menlo.ai/asimov/digital-asimov) robot twin.

```text
route candidates
      -> direct Ising model
      -> CUDA p-bit population annealing
      -> selected route
      -> Digital Asimov PPO locomotion
```

The CUDA solver chooses one of four routes around a predefined obstacle. It
works directly with Ising spins and does **not** construct a QUBO or use a
D-Wave system. The decoded route selects a four-command warehouse movement,
and Digital Asimov's PPO controller executes it beside the table.

Digital Asimov performs those commands through Menlo's existing neural
locomotion policy. The policy was trained with PPO reinforcement learning; this
project uses the trained policy and does not retrain it.

## How it works

Each possible route is represented by a spin with value `-1` or `+1`. The
Ising fields contain route costs, while pair couplings impose the rule that
exactly one route must be selected. The blocked direct path has a large cost,
the safe left detour `BLFL` is the lowest-cost choice, while right and more
conservative detours are more expensive alternatives. `F` is allowed only
after an unused `B`, so forward moves can never outnumber backward moves.

`main.py` launches 4,096 replicas on the GPU. A small CuPy `RawKernel` applies
the probabilistic p-bit update in CUDA. The replicas are resampled with
Boltzmann weights over 40 temperature steps (population annealing).

This follows the general GPU stochastic-spin direction of the
[3-XORSAT paper](https://iopscience.iop.org/article/10.1209/0295-5075/133/60005/meta)
and adds the population method described in the
[GPU population-annealing reference](https://arxiv.org/abs/1703.03676).
Conceptually, it replaces the QUBO, embedding, and D-Wave blocks in the
[inverse-kinematics workflow](https://www.nature.com/articles/s41598-025-34346-z)
with a direct Ising model and a CUDA p-bit solver.

## Setup on Windows PowerShell

The CuPy CUDA 13 wheel includes the required CUDA runtime components, so a
separate `nvcc` installation is not required. An NVIDIA GPU and compatible
driver are still required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a free Menlo account, obtain an API key from the platform settings, and
set it only in the current terminal:

```powershell
$env:MENLO_API_KEY="your-key-here"
$env:MENLO_RCS_URL="https://api.menlo.ai/rcs"
```

Do not put the real key in a source file or commit it to Git.

## Run

```powershell
python main.py
```

The program prints the GPU, Ising result, selected route, and a temporary
Digital Asimov URL. Open the URL in Chrome. After a short policy stabilization
delay, the robot backs away, shifts left, moves forward from the cleared
position, shifts left again, and stops.

## Scope and limitations

- The table is treated as a fixed known obstacle. The demo does not use camera
  perception or dynamic obstacle detection.
- The demo uses Menlo's already-trained PPO locomotion controller rather than
  training a new policy.
- It is an informal end-to-end demonstration, not a performance benchmark or
  a claim that p-bits outperform conventional route planning.

Relevant Menlo references:

- [Python SDK quickstart](https://docs.menlo.ai/platform/get-started/quickstart)
- [Robot control and velocity commands](https://docs.menlo.ai/asimov/1/api/robot-control)
- [PPO locomotion environment](https://docs.menlo.ai/guides/locomotion-training/reinforcement-learning-simulation-training-environment)
