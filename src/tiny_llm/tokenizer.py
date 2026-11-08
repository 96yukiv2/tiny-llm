"""Map individual Unicode characters to integers and back."""


class CharacterTokenizer:
    def __init__(self, characters: str):
        if not characters or len(set(characters)) != len(characters):
            raise ValueError("Vocabulary must contain unique characters and cannot be empty")
        self.characters = characters
        self.ids = {char: index for index, char in enumerate(characters)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        return cls("".join(sorted(set(text))))

    def encode(self, text: str) -> list[int]:
        try:
            return [self.ids[char] for char in text]
        except KeyError as error:
            raise ValueError(f"Character not in training vocabulary: {error.args[0]!r}") from error

    def decode(self, ids: list[int]) -> str:
        if any(index < 0 or index >= len(self.characters) for index in ids):
            raise ValueError("Token ID outside vocabulary")
        return "".join(self.characters[index] for index in ids)
