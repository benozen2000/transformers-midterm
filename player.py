import random
import os
import torch
import chess
from typing import Optional, Dict, List
from transformers import DistilBertForSequenceClassification
from getpass import getpass
from tqdm import tqdm

try:
    from chess_tournament.players import Player as _Base
except ImportError:
    class _Base:
        def __init__(self, name): self.name = name

from chess_tournament import (
    Game, Player, RandomPlayer, LMPlayer, SmolPlayer, EnginePlayer, run_tournament
)


class TransformerPlayer(_Base):
    """
    Chess player using benozen/chess-finetune-output.
    DistilBERT (11M params) trained from scratch on 8M Stockfish depth-16 positions.
    Legal-move masking applied at inference — never returns an illegal move.
    """

    HF_REPO = 'benozen/chess-finetune-output'

    # ── Tokenizer constants ───────────────────────────────────────────────
    _PAD        = 0
    _BOS        = 1
    _EOS        = 2
    _EMPTY      = 3
    _WHITE_MOVE = 16
    _BLACK_MOVE = 17
    _CASTLE_YES = 18
    _CASTLE_NO  = 19
    _NO_EP      = 28
    _SEQ_LEN    = 72
    _VOCAB_SIZE = 29

    _PIECE_TO_ID: Dict[str, int] = {
        '.': 3,
        'P': 4, 'N': 5, 'B': 6, 'R': 7,  'Q': 8,  'K': 9,
        'p':10, 'n':11, 'b':12, 'r':13,  'q':14,  'k':15,
    }

    @classmethod
    def _build_move_index(cls) -> tuple:
        files   = 'abcdefgh'
        promos  = 'qrbn'
        squares = [f + r for f in files for r in '12345678']
        moves: List[str] = []
        for fsq in squares:
            for tsq in squares:
                if fsq != tsq:
                    moves.append(fsq + tsq)
        for fi, ff in enumerate(files):
            for ti, tf in enumerate(files):
                if abs(fi - ti) > 1:
                    continue
                for p in promos:
                    moves.append(ff + '7' + tf + '8' + p)
                    moves.append(ff + '2' + tf + '1' + p)
        move_to_idx = {m: i for i, m in enumerate(moves)}
        idx_to_move = {i: m for i, m in enumerate(moves)}
        return move_to_idx, idx_to_move

    def __init__(self, name: str = 'BenozenChess'):
        super().__init__(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._move_to_idx, self._idx_to_move = self._build_move_index()
        self._num_moves = len(self._move_to_idx)
        self._model = None  # lazy-loaded

    def _encode_fen(self, fen: str) -> List[int]:
        parts    = fen.split(' ')
        board_s  = parts[0]
        side     = parts[1]
        castling = parts[2]
        ep       = parts[3]
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
        print(f'[{self.name}] Loading {self.HF_REPO} on {self.device}...')
        self._model = DistilBertForSequenceClassification.from_pretrained(self.HF_REPO)
        self._model.to(self.device)
        self._model.eval()
        print(f'[{self.name}] Ready.')

    def _predict(self, fen: str) -> Optional[str]:
        board     = chess.Board(fen)
        legal_idx = [self._move_to_idx[m.uci()] for m in board.legal_moves
                     if m.uci() in self._move_to_idx]
        if not legal_idx:
            return None
        ids            = self._encode_fen(fen)
        input_ids      = torch.tensor([ids],     dtype=torch.long).to(self.device)
        attention_mask = torch.ones(1, len(ids), dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids,
                                 attention_mask=attention_mask).logits[0].cpu()
        mask = torch.full((self._num_moves,), float('-inf'))
        for idx in legal_idx:
            mask[idx] = 0.0
        best_idx = int((logits + mask).argmax())
        return self._idx_to_move[best_idx]

    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
            return self._predict(fen)
        except Exception as e:
            print(f'[{self.name}] Error: {e} — falling back to random')
            board = chess.Board(fen)
            moves = list(board.legal_moves)
            return random.choice(moves).uci() if moves else None
