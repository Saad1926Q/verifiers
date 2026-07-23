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


class FifteenPuzzleUser(vf.User[vf.UserConfig, FifteenPuzzleState]):
    async def setup_task(self, task) -> None:
        self.state.current_board = tuple(task.initial_board)
        self.state.moves_taken = []
        self.state.solved = False
        self.state.episode_finished = False
        self.state.illegal_move = False
        self.state.format_failure = False
        self.state.terminal_reason = None

    async def respond(self, message: str) -> vf.Messages:
        move = parse_move(message)

        if move is None:
            self.state.format_failure = True
            self.state.episode_finished = True
            self.state.terminal_reason = "format_failure"
            return []

        if move not in legal_moves(self.state.current_board):
            self.state.illegal_move = True
            self.state.episode_finished = True
            self.state.terminal_reason = "illegal_move"
            return []

        next_board = apply_move(self.state.current_board, move)
        self.state.current_board = next_board
        self.state.moves_taken.append(move)

        if is_solved(next_board):
            self.state.solved = True
            self.state.episode_finished = True
            self.state.terminal_reason = "solved"
            return []

        return [
            {
                "role": "user",
                "content": f"Board after move:\n{render(next_board)}\n\nContinue.",
            }
        ]


if __name__ == "__main__":
    FifteenPuzzleUser.run()
