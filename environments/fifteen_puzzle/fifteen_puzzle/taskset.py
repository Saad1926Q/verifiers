import math

import verifiers.v1 as vf
from fifteen_puzzle.protocol import build_initial_prompt
from fifteen_puzzle.servers.user import (
    FifteenPuzzleState,
    apply_response,
    reset_state,
)

BUCKET_ORDER = ("trivial", "easy", "medium", "hard")


def gaussian_bucket_probs(
    step: int,
    total_steps: int,
    num_buckets: int,
    sigma: float,
    beta: float,
) -> list[float]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < 0 or step > total_steps:
        raise ValueError("step must be between 0 and total_steps")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    progress = step / total_steps

    center = progress**beta * (num_buckets - 1)

    weights = [
        math.exp(-((center - bucket_idx) ** 2) / (2 * sigma**2))
        for bucket_idx in range(num_buckets)
    ]

    total = sum(weights)

    return [weight / total for weight in weights]


def group_rows_by_bucket(rows) -> dict[str, list]:
    grouped = {bucket: [] for bucket in BUCKET_ORDER}

    for row in rows:
        bucket = row["bucket"]
        if bucket not in grouped:
            raise ValueError(f"unknown bucket: {bucket}")
        grouped[bucket].append(row)

    for bucket, bucket_rows in grouped.items():
        if not bucket_rows:
            raise ValueError(f"no rows found for bucket: {bucket}")

    return grouped


class FifteenPuzzleData(vf.TaskData):
    """One row's data. Add task-specific fields here, such as a reference answer."""

    scramble_depth: int
    initial_board: tuple[int, ...]
    optimal_soln: tuple[str, ...]
    optimal_length: int
    bucket: str
    split: str


class FifteenPuzzleTaskConfig(vf.TaskConfig):
    pass


class FifteenPuzzleTask(
    vf.Task[FifteenPuzzleData, FifteenPuzzleState, FifteenPuzzleTaskConfig]
):
    """Rewards, hooks, and servers, with row data available on ``self.data``."""

    @vf.stop
    async def episode_finished(self, trace: vf.Trace) -> bool:
        return trace.state.episode_finished

    @vf.reward(weight=1.0)
    async def reward(self, trace: vf.Trace) -> float:
        if trace.state.solved:
            efficiency = self.data.optimal_length / len(trace.state.moves_taken)
            return 0.8 + 0.2 * min(efficiency, 1.0)

        if trace.state.illegal_move or trace.state.format_failure:
            return -0.1

        return 0.0

    @vf.metric
    async def solved(self, trace: vf.Trace) -> float:
        return float(trace.state.solved)

    @vf.metric
    async def illegal_move(self, trace: vf.Trace) -> float:
        return float(trace.state.illegal_move)

    @vf.metric
    async def format_failure(self, trace: vf.Trace) -> float:
        return float(trace.state.format_failure)

    @vf.metric
    async def num_moves(self, trace: vf.Trace) -> float:
        return float(len(trace.state.moves_taken))

    @vf.metric
    async def efficiency(self, trace: vf.Trace) -> float:
        if not trace.state.solved or not trace.state.moves_taken:
            return 0.0

        return self.data.optimal_length / len(trace.state.moves_taken)


class FifteenPuzzleEnv(vf.SingleAgentEnv):
    async def run(self, task, agents) -> None:
        async with agents.agent.interaction(task) as interaction:
            state = interaction.trace.state
            reset_state(state, task.data)

            segment = await interaction.turn()

            while not segment.terminated and not state.episode_finished:
                messages = apply_response(state, segment.last_reply)

                if state.episode_finished:
                    interaction.trace.stop(state.terminal_reason)
                    break

                segment = await interaction.turn(messages)


class FifteenPuzzleConfig(vf.TasksetConfig):
    dataset_name: str = "saad1926q/15-puzzle"
    subset: str = "rl"
    split: str = "rl"
    curriculum_schedule: str = "none"
    curriculum_step: int = 0
    curriculum_total_steps: int = 3
    steps_per_stage: int | None = None
    batch_size: int | None = None
    group_size: int | None = None
    sigma: float = 0.75
    beta: float = 0.25
    seed: int = 0
    task: FifteenPuzzleTaskConfig = FifteenPuzzleTaskConfig()


class FifteenPuzzleTaskset(vf.Taskset[FifteenPuzzleTask, FifteenPuzzleConfig]):
    def _build_task(self, row, idx: int) -> FifteenPuzzleTask:
        board = tuple(row["board"])
        optimal_soln = tuple(row["optimal_moves"])

        data = FifteenPuzzleData(
            idx=idx,
            prompt=build_initial_prompt(board),
            scramble_depth=row["scramble_depth"],
            initial_board=board,
            optimal_soln=optimal_soln,
            optimal_length=row["optimal_length"],
            bucket=row["bucket"],
            split=row["split"] if "split" in row else self.config.split,
        )

        return FifteenPuzzleTask(data, self.config.task)

    def load(self) -> list[FifteenPuzzleTask]:
        import random

        from datasets import load_dataset

        rows = load_dataset(
            self.config.dataset_name,
            self.config.subset,
            split=self.config.split,
        )

        if self.config.curriculum_schedule == "none":
            return [self._build_task(row, idx) for idx, row in enumerate(rows)]

        if self.config.curriculum_schedule != "gaussian":
            raise ValueError(
                f"unsupported curriculum_schedule: {self.config.curriculum_schedule}"
            )

        rows_by_bucket = group_rows_by_bucket(rows)

        probs = gaussian_bucket_probs(
            step=self.config.curriculum_step,
            total_steps=self.config.curriculum_total_steps,
            num_buckets=len(BUCKET_ORDER),
            sigma=self.config.sigma,
            beta=self.config.beta,
        )

        rng = random.Random(self.config.seed)

        if (
            self.config.steps_per_stage is not None
            and self.config.batch_size is not None
            and self.config.group_size is not None
        ):
            target_count = (
                self.config.steps_per_stage
                * self.config.batch_size
                // self.config.group_size
            )
        else:
            target_count = len(rows)
        tasks = []

        while len(tasks) < target_count:
            bucket = rng.choices(BUCKET_ORDER, weights=probs, k=1)[0]
            row = rng.choice(rows_by_bucket[bucket])
            tasks.append(self._build_task(row, len(tasks)))

        return tasks
