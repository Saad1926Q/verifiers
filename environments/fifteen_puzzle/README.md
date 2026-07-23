# fifteen-puzzle

A v1 `verifiers` environment for the 4x4 Fifteen Puzzle. The model solves a sliding-tile board by emitting one move per turn.

## Task

The initial prompt shows the current board and the solved target:

```text
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 _
```

On each turn, the model must respond with exactly one move wrapped in `<move>...</move>` tags:

```text
<move>left</move>
```

Allowed moves are `up`, `down`, `left`, and `right`. A move describes the numbered tile moving into the blank, not the blank moving. For example, if a row is `13 _ 14 15`, then `<move>left</move>` moves tile `14` left into the blank.

The episode ends when the board is solved, the response format is invalid, or the model makes an illegal move.

## Data

By default, the taskset loads from Hugging Face:

- dataset: `saad1926q/15-puzzle`
- subset: `rl`
- split: `rl`
- tasks: `5`

Each task stores the initial board, scramble depth, optimal solution, optimal length, and split.

## Scoring

Solved puzzles receive:

```text
0.8 + 0.2 * min(optimal_length / moves_taken, 1.0)
```

Invalid format or illegal moves receive `-0.1`. Unsolved non-terminal traces receive `0.0`.

The task reports these metrics:

- `solved`
- `illegal_move`
- `format_failure`
- `num_moves`
- `efficiency`

## Run

Install the environment package and run a small eval:

```bash
uv pip install -e environments/fifteen_puzzle
uv run eval fifteen-puzzle -n 3
```

Useful knobs:

```bash
uv run eval fifteen-puzzle --taskset.num-tasks 10 --model <id> -n 3 -r 1
```

## Layout

- `fifteen_puzzle/taskset.py`: task data, config, reward, metrics, and dataset loading.
- `fifteen_puzzle/protocol.py`: prompt construction and move parsing.
- `fifteen_puzzle/servers/user.py`: puzzle state, turn handling, and terminal conditions.
- `fifteen_puzzle/game/board.py`: board representation, legal moves, and move application.
- `fifteen_puzzle/game/render.py`: board rendering for prompts.
