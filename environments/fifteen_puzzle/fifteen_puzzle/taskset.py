import verifiers.v1 as vf
from fifteen_puzzle.protocol import build_initial_prompt
from fifteen_puzzle.servers.user import (
    FifteenPuzzleState,
    apply_response,
    reset_state,
)


DIFFICULTY_LEVELS: dict[tuple[int, int], str] = {
    (1, 6): "trivial",
    (7, 12): "easy",
    (13, 24): "medium",
    (25, 80): "hard",
}


def difficulty_level(optimal_length: int) -> str:
    for (lower_bound, upper_bound), level in DIFFICULTY_LEVELS.items():
        if lower_bound <= optimal_length <= upper_bound:
            return level

    raise ValueError(f"unsupported optimal_length: {optimal_length}")


class FifteenPuzzleData(vf.TaskData):
    """One row's data. Add task-specific fields here, such as a reference answer."""

    scramble_depth: int
    initial_board: tuple[int, ...]
    optimal_soln: tuple[str, ...]
    optimal_length: int
    difficulty_level: str
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
    num_tasks: int = 5
    """How many tasks to build."""
    task: FifteenPuzzleTaskConfig = FifteenPuzzleTaskConfig()


class FifteenPuzzleTaskset(vf.Taskset[FifteenPuzzleTask, FifteenPuzzleConfig]):
    def load(self) -> list[FifteenPuzzleTask]:
        from datasets import load_dataset

        rows = load_dataset(
            self.config.dataset_name,
            self.config.subset,
            split=self.config.split,
        )

        tasks = []

        for row in rows:
            board = tuple(row["board"])
            optimal_soln = tuple(row["optimal_moves"])

            data = FifteenPuzzleData(
                idx=len(tasks),
                prompt=build_initial_prompt(board),
                scramble_depth=row["scramble_depth"],
                initial_board=board,
                optimal_soln=optimal_soln,
                optimal_length=row["optimal_length"],
                difficulty_level=difficulty_level(row["optimal_length"]),
                split=row["split"] if "split" in row else self.config.split,
            )

            tasks.append(FifteenPuzzleTask(data, self.config.task))

            if len(tasks) >= self.config.num_tasks:
                break

        return tasks
