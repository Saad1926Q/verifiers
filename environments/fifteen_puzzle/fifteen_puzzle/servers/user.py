from pydantic import Field

import verifiers.v1 as vf

from fifteen_puzzle.game.board import apply_move, is_solved, legal_moves
from fifteen_puzzle.game.render import render
from fifteen_puzzle.protocol import parse_move


class FifteenPuzzleState(vf.State):
    current_board: tuple[int, ...] = ()
    moves_taken: list[str] = Field(default_factory=list)
    solved: bool = False
    episode_finished: bool = False
    illegal_move: bool = False
    format_failure: bool = False
    terminal_reason: str | None = None


def reset_state(state: FifteenPuzzleState, task) -> None:
    state.current_board = tuple(task.initial_board)
    state.moves_taken = []
    state.solved = False
    state.episode_finished = False
    state.illegal_move = False
    state.format_failure = False
    state.terminal_reason = None


def apply_response(state: FifteenPuzzleState, message: str) -> vf.Messages:
    move = parse_move(message)

    if move is None:
        state.format_failure = True
        state.episode_finished = True
        state.terminal_reason = "format_failure"
        return []

    if move not in legal_moves(state.current_board):
        state.illegal_move = True
        state.episode_finished = True
        state.terminal_reason = "illegal_move"
        return []

    next_board = apply_move(state.current_board, move)
    state.current_board = next_board
    state.moves_taken.append(move)

    if is_solved(next_board):
        state.solved = True
        state.episode_finished = True
        state.terminal_reason = "solved"
        return []

    return [
        {
            "role": "user",
            "content": f"Board after move:\n{render(next_board)}\n\nContinue.",
        }
    ]
