import re

from fifteen_puzzle.game.render import render

VALID_MOVES = {"up", "down", "left", "right"}
MOVE_PATTERN = re.compile(r"<move>\s*(.*?)\s*</move>", re.IGNORECASE | re.DOTALL)


def parse_move(response_text: str) -> str | None:
    """
    Extract one move tag from the model response and validate its value.
    """

    matches = MOVE_PATTERN.findall(response_text)

    if len(matches) != 1:
        return None

    move = matches[0].strip().lower()

    if move not in VALID_MOVES:
        return None

    return move


def build_system_message() -> dict[str, str]:
    """
    Build the fixed system instruction used for puzzle SFT and RL rollouts.
    """

    return {
        "role": "system",
        "content": (
            "You are a competitive puzzle solver. Make sure you read the puzzle "
            "instructions carefully, and always follow the required format.\n\n"
            "In each turn, think briefly inside <think>...</think> tags, then "
            "output exactly one move inside <move>...</move> tags."
        ),
    }


def build_initial_user_message(board: tuple[int, ...]) -> dict[str, str]:
    """
    Build the initial user message with puzzle rules and the starting board.
    """

    return {
        "role": "user",
        "content": (
            "You are Player 0 in 15-puzzle.\n"
            "A 4x4 sliding puzzle board is given. The blank tile is shown as _.\n"
            "Your goal is to reach the solved board:\n"
            "1 2 3 4\n"
            "5 6 7 8\n"
            "9 10 11 12\n"
            "13 14 15 _\n\n"
            "At each turn, choose one legal move.\n"
            "Moves describe the numbered tile moving into the blank, not the blank moving.\n"
            "For example, if a row is 13 _ 14 15, then <move>left</move> means "
            "tile 14 moves left into the blank, producing 13 14 _ 15.\n\n"
            "Allowed moves are: up, down, left, right.\n"
            "Wrap your move in <move>...</move>, for example: <move>left</move>.\n\n"
            f"Current board:\n{render(board)}"
        ),
    }


def build_initial_prompt(board: tuple[int, ...]) -> list[dict[str, str]]:
    """
    Build the complete initial chat prompt for one puzzle rollout.
    """

    return [
        build_system_message(),
        build_initial_user_message(board),
    ]
