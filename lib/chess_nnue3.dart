import 'chess3.dart';
import 'constants.dart';
import 'nnue_logic_batch2.dart'; // Must export NNUE + NnueAccumulator

class ChessWithNNUE extends Chess {
  // Initialize before super() so overridden load() can safely use them
  final NNUE nnue;
  final NnueAccumulator accumulator;

  ChessWithNNUE()
    : nnue = NNUE(),
      accumulator = NnueAccumulator(NNUE.M),
      super() {
    // Base Chess() ctor already called load(DEFAULT_POSITION)
    // Ensure accumulator matches the current board
    fullNnueRefresh();
  }

  /// Maps 0x88 (0..127) -> 64 (0..63).
  @pragma('vm:prefer-inline')
  int _x88to64(int sq) => ((sq & 7) | ((sq >> 4) << 3));

  /// Convenience wrapper that delegates to the NNUE's HalfKP mapping.
  @pragma('vm:prefer-inline')
  int _halfKP(int k64, int s64, Piece p, bool flip) {
    return nnue.getHalfKPIndex(k64, s64, p, flip);
  }

  /// Completely re-syncs the NNUE accumulator with the current board state.
  void fullNnueRefresh() {
    final wK0x88 = kings[Color.WHITE];
    final bK0x88 = kings[Color.BLACK];

    if (wK0x88 == Chess.EMPTY || bK0x88 == Chess.EMPTY) {
      // No kings on board; keep accumulator at bias only.
      accumulator.refresh(nnue.l0, const <int>[], 0);
      accumulator.refresh(nnue.l0, const <int>[], 1);
      return;
    }

    final whiteKing = _x88to64(wK0x88);
    final blackKing = _x88to64(bK0x88);

    final whiteFeatures = <int>[];
    final blackFeatures = <int>[];

    for (int sq = 0; sq < 128; sq++) {
      if ((sq & 0x88) != 0) continue; // skip padding
      final p = board[sq];
      if (p == null || p.type == PieceType.KING)
        continue; // kings excluded in HalfKP

      final s64 = _x88to64(sq);
      final wIdx = _halfKP(whiteKing, s64, p, false);
      final bIdx = _halfKP(blackKing, s64, p, true);

      if (wIdx != -1) whiteFeatures.add(wIdx);
      if (bIdx != -1) blackFeatures.add(bIdx);
    }

    accumulator.refresh(nnue.l0, whiteFeatures, 0); // white perspective
    accumulator.refresh(nnue.l0, blackFeatures, 1); // black perspective
  }

  /// Auto-sync NNUE on load()/reset()
  @override
  bool load(String fen, {bool check_validity = true}) {
    final ok = super.load(fen, check_validity: check_validity);
    if (ok) fullNnueRefresh();
    return ok;
  }

  @override
  void reset() {
    super.reset();
    fullNnueRefresh();
  }

  double get nnueEvaluation => nnue.evaluate(accumulator, turn);

  @override
  void make_move(Move move) {
    // 1) Apply to board first (king squares update here)
    super.make_move(move);

    // 2) NNUE update
    if (move.piece == PieceType.KING) {
      // Moving a king changes the HalfKP perspective grid
      fullNnueRefresh();
    } else {
      _applyMoveToAccumulator(move, isUndo: false);
    }
  }

  @override
  Move? undo_move() {
    if (history.isEmpty) return null;

    // Peek the move that produced the current state
    final last = history.last.move;
    final wasKingMove = last.piece == PieceType.KING;

    // 1) Revert board
    final undone = super.undo_move();

    // 2) NNUE update (kings are now in the restored position)
    if (wasKingMove) {
      fullNnueRefresh();
    } else {
      _applyMoveToAccumulator(last, isUndo: true);
    }
    return undone;
  }

  /// Play a move given in LAN (Long Algebraic Notation), e.g.:
  /// "e2e4", "e7e8q", "e1g1" (O-O), "e1c1" (O-O-O), etc.
  ///
  /// Returns true if the move was found among legal moves and applied.
  bool moveLAN(String lan) {
    if (lan.isEmpty || lan.length < 4) return false;

    final from = lan.substring(0, 2);
    final to = lan.substring(2, 4);

    // Optional promotion trailing char (q,r,b,n), case-insensitive.
    PieceType? promo;
    if (lan.length > 4) {
      promo = _promoFromChar(lan[4]);
      if (promo == null) return false; // invalid promo piece
    }

    // Generate legal moves from the current position
    final moves = generate_moves();
    if (moves.isEmpty) return false;

    // Try to match by coordinates + optional promotion
    for (final m in moves) {
      if (m.fromAlgebraic == from && m.toAlgebraic == to) {
        final isPromotionMove = (m.flags & BITS_PROMOTION) != 0;

        if (isPromotionMove) {
          // If user didn't specify promo, or specified a different one — skip
          if (promo == null || m.promotion != promo) continue;
        } else {
          // Non-promotion moves must not have a promo char in LAN
          if (promo != null) continue;
        }

        // Found a matching legal move — play it
        make_move(m);
        return true;
      }
    }

    // Fallback: handle castling by SAN if needed (rare)
    if ((from == 'e1' && to == 'g1') || (from == 'e8' && to == 'g8')) {
      return move("O-O");
    }
    if ((from == 'e1' && to == 'c1') || (from == 'e8' && to == 'c8')) {
      return move("O-O-O");
    }

    return false;
  }

  /// Map a promotion character to PieceType.
  PieceType? _promoFromChar(String ch) {
    switch (ch.toLowerCase()) {
      case 'q':
        return PieceType.QUEEN;
      case 'r':
        return PieceType.ROOK;
      case 'b':
        return PieceType.BISHOP;
      case 'n':
        return PieceType.KNIGHT;
      default:
        return null;
    }
  }

  /// Incremental accumulator update for non-king moves.
  /// IMPORTANT: We compute indices using CURRENT king squares (which are
  /// unchanged across non-king moves).
  void _applyMoveToAccumulator(Move move, {required bool isUndo}) {
    final us = move.color;
    final them = (us == Color.WHITE) ? Color.BLACK : Color.WHITE;

    final wK0x88 = kings[Color.WHITE];
    final bK0x88 = kings[Color.BLACK];
    if (wK0x88 == Chess.EMPTY || bK0x88 == Chess.EMPTY) {
      fullNnueRefresh();
      return;
    }

    final wKing = _x88to64(wK0x88);
    final bKing = _x88to64(bK0x88);

    final addedW = <int>[];
    final removedW = <int>[];
    final addedB = <int>[];
    final removedB = <int>[];

    final from64 = _x88to64(move.from);
    final to64 = _x88to64(move.to);

    void stageChange(int sq64, PieceType type, Color color, bool isAddition) {
      final idxW = _halfKP(wKing, sq64, Piece(type, color), false);
      final idxB = _halfKP(bKing, sq64, Piece(type, color), true);

      if (idxW != -1) {
        if (isAddition)
          addedW.add(idxW);
        else
          removedW.add(idxW);
      }
      if (idxB != -1) {
        if (isAddition)
          addedB.add(idxB);
        else
          removedB.add(idxB);
      }
    }

    // Moving piece: remove from 'from', add at 'to' (promotion handled)
    stageChange(from64, move.piece, us, false);
    if (move.promotion != null) {
      stageChange(to64, move.promotion!, us, true);
    } else {
      stageChange(to64, move.piece, us, true);
    }

    // Captures: remove captured piece at capture square
    if (move.captured != null) {
      int capSq64 = to64;
      if ((move.flags & BITS_EP_CAPTURE) != 0) {
        final cap0x88 = (us == Color.WHITE) ? (move.to + 16) : (move.to - 16);
        capSq64 = _x88to64(cap0x88);
      }
      stageChange(capSq64, move.captured!, them, false);
    }

    // Castling rook displacement (this block rarely triggers here because king moves do full refresh)
    if ((move.flags & (BITS_KSIDE_CASTLE | BITS_QSIDE_CASTLE)) != 0) {
      int rFrom, rTo;
      if ((move.flags & BITS_KSIDE_CASTLE) != 0) {
        rFrom = move.to + 1;
        rTo = move.to - 1;
      } else {
        rFrom = move.to - 2;
        rTo = move.to + 1;
      }
      stageChange(_x88to64(rFrom), PieceType.ROOK, us, false);
      stageChange(_x88to64(rTo), PieceType.ROOK, us, true);
    }

    // Apply to accumulator (swap add/remove if undo)
    if (isUndo) {
      accumulator.update(nnue.l0, removedW, addedW, 0);
      accumulator.update(nnue.l0, removedB, addedB, 1);
    } else {
      accumulator.update(nnue.l0, addedW, removedW, 0);
      accumulator.update(nnue.l0, addedB, removedB, 1);
    }
  }
}

void main() {
  print("Initializing ChessWithNNUE Engine...");

  final game = ChessWithNNUE();

  print("Initial Position Eval: ${game.nnueEvaluation.toStringAsFixed(4)}");

  print("\nMoving: 1. e4");
  game.move("e4");
  print("Eval after 1. e4: ${game.nnueEvaluation.toStringAsFixed(4)}");

  print("Moving: 1... e5");
  game.move("e5");
  print("Eval after 1... e5: ${game.nnueEvaluation.toStringAsFixed(4)}");

  // Load a tactical position and refresh via load()
  game.load("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");

  print("\nPosition: Knight can capture on e5");
  print("Before capture: ${game.nnueEvaluation.toStringAsFixed(4)}");

  game.move("Nxe5");
  print("After Nxe5: ${game.nnueEvaluation.toStringAsFixed(4)}");

  print("\nCurrent Turn: ${game.turn == Color.WHITE ? 'White' : 'Black'}");
}
