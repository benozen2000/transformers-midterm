import subprocess
import sys

def _pip(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

try:
    import torch
except ImportError:
    _pip('torch')
    import torch

try:
    import chess
except ImportError:
    _pip('chess')
    import chess

try:
    from transformers import DistilBertForSequenceClassification
except ImportError:
    _pip('transformers')
    from transformers import DistilBertForSequenceClassification

import random
from typing import Optional, Dict, List

try:
    from chess_tournament.players import Player as _Base
except ImportError:
    class _Base:
        def __init__(self, name): self.name = name


class TransformerPlayer(_Base):

    HF_REPO = 'benozen/chess-finetune-output'

    _BOS        = 1
    _EOS        = 2
    _EMPTY      = 3
    _WHITE_MOVE = 16
    _BLACK_MOVE = 17
    _CASTLE_YES = 18
    _CASTLE_NO  = 19
    _NO_EP      = 28

    _PIECE_TO_ID: Dict[str, int] = {
        'P': 4, 'N': 5, 'B': 6, 'R': 7,  'Q': 8,  'K': 9,
        'p':10, 'n':11, 'b':12, 'r':13,  'q':14,  'k':15,
    }

    @classmethod
    def _build_move_index(cls) -> tuple:
        files   = 'abcdefgh'
        squares = [f + r for f in files for r in '12345678']
        moves: List[str] = [fsq + tsq for fsq in squares for tsq in squares if fsq != tsq]
        for fi, ff in enumerate(files):
            for ti, tf in enumerate(files):
                if abs(fi - ti) <= 1:
                    for p in 'qrbn':
                        moves += [ff + '7' + tf + '8' + p, ff + '2' + tf + '1' + p]
        return {m: i for i, m in enumerate(moves)}, {i: m for i, m in enumerate(moves)}

    def __init__(self, name: str = 'BenozenChess'):
        super().__init__(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._move_to_idx, self._idx_to_move = self._build_move_index()
        self._num_moves = len(self._move_to_idx)
        self._model = None

    def _encode_fen(self, fen: str) -> List[int]:
        board_s, side, castling, ep = fen.split(' ')[:4]
        ids = [self._BOS]
        for ch in board_s:
            if ch.isdigit():
                ids.extend([self._EMPTY] * int(ch))
            elif ch != '/':
                ids.append(self._PIECE_TO_ID.get(ch, self._EMPTY))
        ids.append(self._WHITE_MOVE if side == 'w' else self._BLACK_MOVE)
        for flag in 'KQkq':
            ids.append(self._CASTLE_YES if flag in castling else self._CASTLE_NO)
        ids.append(self._NO_EP if ep == '-' else 20 + ord(ep[0]) - ord('a'))
        ids.append(self._EOS)
        return ids

    def _load_model(self):
        if self._model is not None:
            return
        self._model = DistilBertForSequenceClassification.from_pretrained(self.HF_REPO)
        self._model.to(self.device)
        self._model.eval()

    def _predict(self, fen: str) -> Optional[str]:
        board     = chess.Board(fen)
        legal_idx = [self._move_to_idx[m.uci()] for m in board.legal_moves
                     if m.uci() in self._move_to_idx]
        if not legal_idx:
            return None
        ids  = self._encode_fen(fen)
        with torch.no_grad():
            logits = self._model(
                input_ids      = torch.tensor([ids],     dtype=torch.long).to(self.device),
                attention_mask = torch.ones(1, len(ids), dtype=torch.long).to(self.device),
            ).logits[0].cpu()
        mask = torch.full((self._num_moves,), float('-inf'))
        for idx in legal_idx:
            mask[idx] = 0.0
        return self._idx_to_move[int((logits + mask).argmax())]

    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
            return self._predict(fen)
        except Exception:
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            return random.choice(moves).uci() if moves else None
