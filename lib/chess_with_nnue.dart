// lib/chess_with_nnue.dart
import 'chess3.dart';
import 'constants.dart';
import 'nnue_reference.dart';

/// Chess engine wrapper that embeds the NNUERef core and keeps an accumulator
/// in sync with the board (full refresh on king moves; incremental otherwise).
class ChessWithNNUE extends Chess {
  /// NNUE evaluator (float reference implementation).
  final NNUERef nnue;

  /// Two-perspective accumulator (white/black).
  final NnueAccumulatorRef accumulator;

  ChessWithNNUE()
    : nnue = NNUERef(),
      accumulator = NnueAccumulatorRef(NNUERef.M),
      super() {
    // Base Chess() ctor already called load(DEFAULT_POSITION),
    // so we must sync the accumulator now.
    fullNnueRefresh();
  }

  /// Rebuild both perspectives (white & black) accumulators from the board.
  void fullNnueRefresh() {
    nnue.refreshAccumulator(accumulator, board);
  }

  /// Auto-sync NNUE on load()/reset().
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

  /// NNUE evaluation in centipawns from the **side-to-move** perspective.
  double get nnueEvaluation => nnue.evaluate(accumulator, turn);

  /// Make a move and incrementally update the accumulator.
  /// If a king moves, we do a full refresh (the king participates in all features).
  @override
  void make_move(Move move) {
    // Snapshot the board BEFORE the move for delta computation
    final boardBefore = List<Piece?>.from(board);
    final wasKingMove = (move.piece == PieceType.KING);

    // Apply the move on the base engine (board changes here).
    super.make_move(move);

    if (wasKingMove) {
      // King moved => all features bound to king square change => full refresh.
      fullNnueRefresh();
      return;
    }

    // Compute feature deltas using the "before" position and the move info.
    final delta = nnue.deltaForMove(boardBefore: boardBefore, move: move);

    // Defensive fallback in case of future refactors.
    final addedW = delta['addedW'] ?? const <int>[];
    final removedW = delta['removedW'] ?? const <int>[];
    final addedB = delta['addedB'] ?? const <int>[];
    final removedB = delta['removedB'] ?? const <int>[];

    // Apply deltas to both perspectives (white = 0, black = 1).
    accumulator.update(nnue.l0, addedW, removedW, 0);
    accumulator.update(nnue.l0, addedB, removedB, 1);
  }

  /// Undo a move and revert the accumulator incrementally.
  /// For king moves, do a full refresh (kings affect all features).
  @override
  Move? undo_move() {
    if (history.isEmpty) return null;

    // The forward move that produced the current state (to be undone).
    final fwdMove = history.last.move;
    final wasKingMove = (fwdMove.piece == PieceType.KING);

    // 1) Revert the board to the state before fwdMove.
    final undone = super.undo_move();

    // 2) Sync NNUE.
    if (wasKingMove) {
      fullNnueRefresh();
      return undone;
    }

    // We are now back at "boardBefore". Compute delta for the forward move,
    // then invert add/remove to revert the accumulator.
    final boardBefore = List<Piece?>.from(board);
    final delta = nnue.deltaForMove(boardBefore: boardBefore, move: fwdMove);

    final addedW = delta['addedW'] ?? const <int>[];
    final removedW = delta['removedW'] ?? const <int>[];
    final addedB = delta['addedB'] ?? const <int>[];
    final removedB = delta['removedB'] ?? const <int>[];

    // Invert: (added <-> removed)
    accumulator.update(nnue.l0, removedW, addedW, 0);
    accumulator.update(nnue.l0, removedB, addedB, 1);
    return undone;
  }

  // --------------------------
  // Optional: moveLAN helper
  // --------------------------

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
          if (promo == null || m.promotion != promo) continue;
        } else {
          if (promo != null) continue;
        }

        // Found a matching legal move — play it (this triggers NNUE updates above)
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
}
