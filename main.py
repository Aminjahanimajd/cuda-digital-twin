"""Small CUDA p-bit demo for the Digital Asimov robot."""

import asyncio
import os
import time

import cupy as cp
from menlo_robot_sdk import AsyncClient, connect
from menlo_robot_sdk.experimental import generate_room_key


# F = forward, L = turn left, R = turn right.
# The direct route is expensive because it crosses the imaginary obstacle.
ROUTES = [
    {"name": "direct (blocked by table)", "cost": 22.0, "actions": "F"},
    {"name": "safe left detour", "cost": 5.0, "actions": "BLFL"},
    {"name": "safe right detour", "cost": 5.4, "actions": "BRFR"},
    {"name": "conservative detour", "cost": 7.0, "actions": "BBLF"},
]

COMMANDS = {
    "F": {"vx": 0.8, "vy": 0.0, "wz": 0.0, "duration_s": 1.0},
    "L": {"vx": 0.0, "vy": 0.8, "wz": 0.0, "duration_s": 1.0},
    "R": {"vx": 0.0, "vy": -0.8, "wz": 0.0, "duration_s": 1.0},
    "B": {"vx": -0.8, "vy": 0.0, "wz": 0.0, "duration_s": 1.0},
}

REPLICAS = 4096
PENALTY = 30.0


# One CUDA thread updates one replica of one p-bit.
KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__
    void update(signed char *s, float *h, float *J, float *rnd,
                int bit, int n, int replicas, float beta) {
        int r = blockDim.x * blockIdx.x + threadIdx.x;
        if (r >= replicas) return;

        float local = h[bit];
        for (int j = 0; j < n; j++)
            local += J[bit*n + j] * s[r*n + j];

        float probability_plus = 1.0f / (1.0f + expf(2.0f*beta*local));
        s[r*n + bit] = rnd[r] < probability_plus ? 1 : -1;
    }
    """,
    "update",
)


def solve_route():
    """Choose one route with CUDA p-bits and population annealing."""
    cp.random.seed(7)
    costs = cp.array([route["cost"] for route in ROUTES], dtype=cp.float32)
    n = len(ROUTES)

    # Direct Ising form of: cost*x + PENALTY*(sum(x)-1)^2, x=(s+1)/2.
    h = costs / 2 + PENALTY * (n - 2) / 2
    J = cp.full((n, n), PENALTY / 2, dtype=cp.float32)
    cp.fill_diagonal(J, 0)

    spins = cp.where(
        cp.random.random((REPLICAS, n)) < 0.5, cp.int8(-1), cp.int8(1)
    )

    def energies():
        return cp.sum(spins * h, axis=1) + 0.5 * cp.sum((spins @ J) * spins, axis=1)

    betas = cp.linspace(0.1, 5.0, 40)
    old_beta = float(betas[0])

    for step, beta_value in enumerate(betas):
        beta = float(beta_value)

        # Population annealing: favor low-energy replicas as beta increases.
        if step:
            e = energies()
            weights = cp.exp(-(beta - old_beta) * (e - e.min()))
            weights /= weights.sum()
            spins = spins[cp.random.choice(REPLICAS, REPLICAS, p=weights)].copy()

        # Two p-bit sweeps. Each bit is sequential; replicas run in parallel.
        for _ in range(2):
            for bit in range(n):
                rnd = cp.random.random(REPLICAS, dtype=cp.float32)
                KERNEL(
                    ((REPLICAS + 255) // 256,),
                    (256,),
                    (spins, h, J, rnd, bit, n, REPLICAS, cp.float32(beta)),
                )
        old_beta = beta

    cp.cuda.Stream.null.synchronize()
    selected = (spins.astype(cp.int16) + 1) // 2
    valid = cp.where(selected.sum(axis=1) == 1)[0]
    if valid.size == 0:
        raise RuntimeError("No valid route was found")

    # The best valid sample has the smallest original route cost.
    valid_costs = selected[valid] @ costs
    best = valid[cp.argmin(valid_costs)]
    route_number = int(cp.argmax(selected[best]).item())
    return ROUTES[route_number], float(valid_costs.min()), cp.asnumpy(spins[best]).tolist()


async def wait_for_simulator(session):
    print("Waiting for the browser simulator...")
    end = time.monotonic() + 180
    while time.monotonic() < end:
        try:
            if await session.discover_skills():
                print("Simulator connected.")
                return
        except (RuntimeError, TimeoutError):
            pass
        await asyncio.sleep(2)
    raise TimeoutError("Open the printed simulator URL in Chrome")


async def run_robot(route):
    """Execute one simple table-avoidance movement."""
    session = None
    robot_id = None
    stop = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration_s": 0.25}

    async with AsyncClient() as client:
        try:
            robot = await client.robots.create(name="pbit-demo", model="asimov-v0")
            robot_id = robot.robot.id
            session = await connect(
                client,
                robot_id,
                worker_names=[],
                rcw_identity_prefix="simplesim",
                join_livekit=True,
            )
            key = await generate_room_key(client, robot_id)
            print(f"\nOpen in Chrome:\nhttps://sim.menlo.ai/?key={key}\n")
            await wait_for_simulator(session)

            print("Waiting 15 seconds for the walking policy to stabilize...")
            await asyncio.sleep(15)

            actions = route["actions"]
            print("Executing table-avoidance actions:", actions)
            print("L=left, R=right, B=backward, F=toward the table")
            forward_credit = 0
            lost_replies = 0
            for action in actions:
                if action == "B":
                    forward_credit += 1
                elif action == "F":
                    if forward_credit == 0:
                        raise RuntimeError("Forward movement requires an earlier B")
                    forward_credit -= 1

                try:
                    result = await session.invoke(
                        "set_velocity", COMMANDS[action], timeout_s=15
                    )
                    print(action, result.status, "forward credit:", forward_credit)
                    if result.status != "done":
                        raise RuntimeError(f"Movement failed: {result.error}")
                except TimeoutError:
                    lost_replies += 1
                    print(action, "sent, but its reply was lost; continuing safely")
                await asyncio.sleep(1.5)

            try:
                await session.invoke("set_velocity", stop, timeout_s=15)
            except TimeoutError:
                lost_replies += 1
                print("Stop command reply was lost.")

            if lost_replies:
                print("Route finished with", lost_replies, "lost simulator reply/replies.")
            else:
                print("Task complete. Robot executed the safe detour and stopped.")
        finally:
            if session:
                try:
                    await session.invoke("set_velocity", stop, timeout_s=15)
                except Exception:
                    pass
                await session.disconnect()
            if robot_id:
                await client.robots.delete(robot_id)


def main():
    if not os.getenv("MENLO_API_KEY"):
        raise SystemExit("Set MENLO_API_KEY before running this program")
    os.environ.setdefault("MENLO_RCS_URL", "https://api.menlo.ai/rcs")

    device = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device, bytes):
        device = device.decode()
    print("GPU:", device)

    route, energy, spins = solve_route()
    print("Final spins:", spins)
    print("Final energy:", energy)
    print("Selected route:", route["name"])
    asyncio.run(run_robot(route))


if __name__ == "__main__":
    main()
