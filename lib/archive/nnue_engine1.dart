import 'dart:math';
import '../engine/chess.dart';
import '../engine/chess_with_nnue.dart';
import '../engine/constants.dart';
import '../engine/nnue_reference.dart';
import '../engine/tt.dart';

const double INF = 1e12;

/// ===========================================================
///   UNIVERSAL STATIC NNUE BOARD EVALUATOR
/// ===========================================================
extension NNUEBoardEval on NNUERef {
  double evaluateBoard(List<Piece?> board, Color turn) {
    final acc = newAccumulator();
    refreshAccumulator(acc, board); // full rebuild
    return evaluate(acc, turn);
  }
}

/// ===========================================================
///   ROOT MOVE CLASS  (score + move + helpers)
/// ===========================================================
class RootMove {
  final Move move;
  double score;

  RootMove(this.move, this.score);

  @override
  String toString() => "${move.fromAlgebraic}${move.toAlgebraic} ($score)";
}

/// ===========================================================
///   NEGAMAX + ALPHA-BETA PRUNING
/// ===========================================================
/// color = +1 (side to move), -1 (opponent)
double negamax(
  ChessWithNNUE pos,
  int depth,
  double alpha,
  double beta,
  int color,
  TranspositionTable tt,
) {
  final int hash = pos.zobristKey; // MUST use zobrist hash from your engine

  // --- Probe Transposition Table ---
  // --- Probe Transposition Table ---
  final HashEntry? entry = tt.probe(hash);

  const bool debugTT = false; // set true to enable TT debug logs

  if (debugTT) {
    if (entry == null) {
      print("TT MISS  hash=$hash");
    } else {
      print(
        "TT HIT   hash=$hash  "
        "storedDepth=${entry.depth}  searchDepth=$depth  "
        "bound=${entry.bound}  score=${entry.score}",
      );
    }
  }

  if (entry != null && entry.depth >= depth) {
    switch (entry.bound) {
      case Bound.EXACT:
        if (debugTT) print("TT EXACT → return ${entry.score}");
        return entry.score;

      case Bound.LOWER:
        if (debugTT) print("TT LOWER → alpha=$alpha → ${entry.score}");
        alpha = max(alpha, entry.score);
        break;

      case Bound.UPPER:
        if (debugTT) print("TT UPPER → beta=$beta → ${entry.score}");
        beta = min(beta, entry.score);
        break;
    }

    if (alpha >= beta) {
      if (debugTT) {
        print("TT CUTOFF  alpha=$alpha beta=$beta  → return ${entry.score}");
      }
      return entry.score;
    }
  }

  // Leaf or terminal
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
  if (moves.isEmpty) {
    final eval = color * pos.nnue.evaluateBoard(pos.board, pos.turn);
    tt.store(
      HashEntry(hash: hash, score: eval, depth: depth, bound: Bound.EXACT),
    );
    return eval;
  }

  for (final mv in moves) {
    pos.make_move(mv);

    double score = -negamax(pos, depth - 1, -beta, -alpha, -color, tt);

    pos.undo_move();

    if (score > best) {
      best = score;
      bestMove = mv;
    }

    alpha = max(alpha, score);
    if (alpha >= beta) break;
  }

  // --- Store into TT ---
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

/// ===========================================================
///   ROOT NEGAMAX SEARCH (one depth)
/// ===========================================================
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

/// ===========================================================
///   ITERATIVE DEEPENING DRIVER
/// ===========================================================
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

/// ===========================================================
///   MAIN ENGINE WRAPPER
/// ===========================================================
class NNUEEngine extends ChessWithNNUE {
  NNUEEngine() : super();
  NNUEEngine.fromFEN(String fen) : super() {
    load(fen);
  }

  double eval() => nnue.evaluateBoard(board, turn);

  /// Choose and play the best move using iterative deepening + negamax + alpha-beta
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

  /// LAN interface
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
