import 'dart:math';
import '../engine/chess.dart';
import '../engine/chess_with_nnue.dart';
import '../engine/constants.dart';
import '../engine/nnue_reference.dart';

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
) {
  // leaf or terminal
  if (depth == 0 || pos.in_checkmate) {
    return color * pos.nnue.evaluateBoard(pos.board, pos.turn);
  }

  final moves = pos.generate_moves();
  if (moves.isEmpty) {
    return color * pos.nnue.evaluateBoard(pos.board, pos.turn);
  }

  double best = -INF;

  for (final mv in moves) {
    pos.make_move(mv);

    double score = -negamax(pos, depth - 1, -beta, -alpha, -color);

    pos.undo_move();

    if (score > best) best = score;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break; // alpha-beta cutoff
  }

  return best;
}

/// ===========================================================
///   ROOT NEGAMAX SEARCH (one depth)
/// ===========================================================
RootMove rootSearch(ChessWithNNUE pos, int depth) {
  final moves = pos.generate_moves();
  if (moves.isEmpty) {
    return RootMove(null as Move, pos.nnue.evaluateBoard(pos.board, pos.turn));
  }

  double alpha = -INF;
  double beta = INF;

  RootMove? bestRM;
  final rootMoves = <RootMove>[];

  for (final mv in moves) {
    pos.make_move(mv);

    double score = -negamax(pos, depth - 1, -beta, -alpha, -1);

    pos.undo_move();

    rootMoves.add(RootMove(mv, score));

    if (score > alpha) {
      alpha = score;
      bestRM = rootMoves.last;
    }
  }

  // Sort moves by score (descending)
  rootMoves.sort((a, b) => b.score.compareTo(a.score));

  print("info depth $depth");
  for (final rm in rootMoves.take(5)) {
    print(
      "  move ${rm.move.fromAlgebraic}${rm.move.toAlgebraic} score ${rm.score}",
    );
  }

  return bestRM!;
}

/// ===========================================================
///   ITERATIVE DEEPENING DRIVER
/// ===========================================================
RootMove iterativeDeepening(ChessWithNNUE pos, int maxDepth) {
  RootMove? best;

  for (int depth = 1; depth <= maxDepth; depth++) {
    best = rootSearch(pos, depth);

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
