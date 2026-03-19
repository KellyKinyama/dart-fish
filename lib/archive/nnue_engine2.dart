import 'dart:math';
import '../engine/chess.dart';
import '../engine/chess_with_nnue.dart';
import '../engine/constants.dart';
import '../engine/nnue_reference.dart';
import '../engine/tt.dart';

const double INF = 1e12;
const double FUTILITY_MARGIN = 150.0;

// ===========================================================
//   UNIVERSAL STATIC NNUE BOARD EVALUATOR
// ===========================================================
extension NNUEBoardEval on NNUERef {
  double evaluateBoard(List<Piece?> board, Color turn) {
    final acc = newAccumulator();
    refreshAccumulator(acc, board); // full rebuild
    return evaluate(acc, turn);
  }
}

// ===========================================================
//   ROOT MOVE CLASS
// ===========================================================
class RootMove {
  final Move move;
  double score;

  RootMove(this.move, this.score);

  @override
  String toString() => "${move.fromAlgebraic}${move.toAlgebraic} ($score)";
}

// ===========================================================
//   HELPERS
// ===========================================================
bool isQuiet(Move mv) {
  return (mv.flags & (BITS_CAPTURE | BITS_PROMOTION | BITS_EP_CAPTURE)) == 0;
}

// ===========================================================
//   NEGAMAX + ALPHA-BETA + TT + FUTILITY + LMR
// ===========================================================
double negamax(
  ChessWithNNUE pos,
  int depth,
  double alpha,
  double beta,
  int color,
  TranspositionTable tt,
) {
  final int hash = pos.zobristKey;

  // ----------------------------------------------------------
  // TT PROBE
  // ----------------------------------------------------------
  final HashEntry? entry = tt.probe(hash);

  if (entry != null && entry.depth >= depth) {
    switch (entry.bound) {
      case Bound.EXACT:
        return entry.score;
      case Bound.LOWER:
        alpha = max(alpha, entry.score);
        break;
      case Bound.UPPER:
        beta = min(beta, entry.score);
        break;
    }
    if (alpha >= beta) return entry.score;
  }

  // ----------------------------------------------------------
  // Terminal or depth 0
  // ----------------------------------------------------------
  if (depth == 0 || pos.in_checkmate) {
    final eval = color * pos.nnue.evaluateBoard(pos.board, pos.turn);

    tt.store(
      HashEntry(hash: hash, score: eval, depth: depth, bound: Bound.EXACT),
    );
    return eval;
  }

  double best = -INF;
  Move? bestMove;

  final moves = pos.generate_moves();

  // No moves = terminal
  if (moves.isEmpty) {
    final eval = color * pos.nnue.evaluateBoard(pos.board, pos.turn);
    tt.store(
      HashEntry(hash: hash, score: eval, depth: depth, bound: Bound.EXACT),
    );
    return eval;
  }

  int moveIndex = 0;

  // ----------------------------------------------------------
  // Main move loop
  // ----------------------------------------------------------
  for (final mv in moves) {
    pos.make_move(mv);

    // ========================================================
    // FUTILITY PRUNING (only at depth 1, quiet moves, no check)
    // ========================================================
    if (depth == 1 && !pos.in_check && isQuiet(mv)) {
      double staticEval = color * pos.nnue.evaluateBoard(pos.board, pos.turn);

      if (staticEval + FUTILITY_MARGIN <= alpha) {
        pos.undo_move();
        moveIndex++;
        continue; // prune this child
      }
    }

    // ========================================================
    // LMR (Late Move Reductions)
    // ========================================================
    int newDepth = depth - 1;

    bool quiet = isQuiet(mv);

    if (depth >= 3 && moveIndex > 0 && !pos.in_check && quiet) {
      int reduction = 1;
      newDepth = max(1, newDepth - reduction);
    }

    // --------------------------------------------------------
    // Child search
    // --------------------------------------------------------
    double score = -negamax(pos, newDepth, -beta, -alpha, -color, tt);

    pos.undo_move();

    if (score > best) {
      best = score;
      bestMove = mv;
    }

    alpha = max(alpha, score);
    if (alpha >= beta) break; // αβ cutoff

    moveIndex++;
  }

  // ----------------------------------------------------------
  // TT STORE
  // ----------------------------------------------------------
  Bound bound;
  if (best <= alpha) {
    bound = Bound.UPPER;
  } else if (best >= beta) {
    bound = Bound.LOWER;
  } else {
    bound = Bound.EXACT;
  }

  tt.store(
    HashEntry(
      hash: hash,
      score: best,
      depth: depth,
      move: bestMove,
      bound: bound,
    ),
  );

  return best;
}

// ===========================================================
//   ROOT NEGAMAX SEARCH
// ===========================================================
RootMove rootSearch(ChessWithNNUE pos, int depth, TranspositionTable tt) {
  final moves = pos.generate_moves();
  if (moves.isEmpty) {
    return RootMove(null as Move, pos.nnue.evaluateBoard(pos.board, pos.turn));
  }

  double alpha = -INF;
  double beta = INF;

  RootMove? bestRM;

  for (final mv in moves) {
    pos.make_move(mv);

    double score = -negamax(pos, depth - 1, -beta, -alpha, -1, tt);

    pos.undo_move();

    if (score > alpha) {
      alpha = score;
      bestRM = RootMove(mv, score);
    }
  }

  return bestRM!;
}

// ===========================================================
//   ITERATIVE DEEPENING DRIVER
// ===========================================================
RootMove iterativeDeepening(ChessWithNNUE pos, int maxDepth) {
  final tt = TranspositionTable();
  RootMove? best;

  for (int depth = 1; depth <= maxDepth; depth++) {
    best = rootSearch(pos, depth, tt);

    print(
      "info depth $depth selmove ${best.move.fromAlgebraic}${best.move.toAlgebraic} score ${best.score}",
    );
  }

  return best!;
}

// ===========================================================
//   MAIN ENGINE WRAPPER
// ===========================================================
class NNUEEngine extends ChessWithNNUE {
  NNUEEngine() : super();
  NNUEEngine.fromFEN(String fen) : super() {
    load(fen);
  }

  double eval() => nnue.evaluateBoard(board, turn);

  String play({int depth = 4}) {
    final best = iterativeDeepening(this, depth);
    final mv = best.move;

    make_move(mv);

    final last = history.last.move;
    String lan = last.fromAlgebraic + last.toAlgebraic;

    if (last.promotion != null) {
      lan += last.promotion!.name[0].toLowerCase();
    }
    return lan;
  }

  bool moveLAN(String lan) {
    if (lan.length < 4) return false;

    final from = lan.substring(0, 2);
    final to = lan.substring(2, 4);

    PieceType? promo;
    if (lan.length > 4) promo = _promo(lan[4]);

    final moves = generate_moves();
    for (final m in moves) {
      if (m.fromAlgebraic == from && m.toAlgebraic == to) {
        if ((m.flags & BITS_PROMOTION) != 0) {
          if (promo == null || promo != m.promotion) continue;
        } else {
          if (promo != null) continue;
        }
        make_move(m);
        return true;
      }
    }
    return false;
  }

  PieceType? _promo(String ch) {
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
