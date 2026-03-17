import 'dart:math';
import 'chess3.dart';
import 'chess_with_nnue.dart';
import 'constants.dart';

const double INF = 1e12;

/// Minimax + NNUE evaluation
List minimaxNnue(ChessWithNNUE game, int depth, bool maximizingPlayer) {
  // Terminal node?
  if (depth == 0 || game.in_checkmate) {
    return [game.nnueEvaluation, null];
  }

  final moves = game.generate_moves();
  if (moves.isEmpty) {
    // stalemate or no legal moves
    return [game.nnueEvaluation, null];
  }

  if (maximizingPlayer) {
    double bestScore = -INF;
    var bestMove;

    for (final mv in moves) {
      game.make_move(mv);

      double score = minimaxNnue(game, depth - 1, false)[0];

      game.undo_move();

      if (score > bestScore) {
        bestScore = score;
        bestMove = mv;
      }
    }

    return [bestScore, bestMove];
  } else {
    double bestScore = INF;
    var bestMove;

    for (final mv in moves) {
      game.make_move(mv);

      double score = minimaxNnue(game, depth - 1, true)[0];

      game.undo_move();

      if (score < bestScore) {
        bestScore = score;
        bestMove = mv;
      }
    }

    return [bestScore, bestMove];
  }
}

class NNUEEngine extends ChessWithNNUE {
  NNUEEngine() : super();
  NNUEEngine.fromFEN(String fen) : super() {
    load(fen);
  }

  /// NNUE eval wrapper (already centipawns)
  double eval() => nnueEvaluation;

  /// Choose and play the best move using NNUE + minimax depth 2
  String play({int depth = 2}) {
    final best = minimaxNnue(this, depth, true);
    final mv = best[1];

    if (mv == null) return "";

    make_move(mv);

    final last = history.last.move;
    String lan = last.fromAlgebraic + last.toAlgebraic;

    if (last.promotion != null) {
      lan += last.promotion!.name[0].toLowerCase();
    }

    return lan;
  }

  /// LAN input interface, same as your original engine
  bool moveLAN(String lan) {
    if (lan.length < 4) return false;

    final from = lan.substring(0, 2);
    final to = lan.substring(2, 4);

    PieceType? promo;
    if (lan.length > 4) {
      promo = _promoFromChar(lan[4]);
      if (promo == null) return false;
    }

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

void main() {
  final engine = NNUEEngine();

  print("NNUE evaluation of start position: ${engine.eval()}");

  // Play best move with NNUE minimax
  final mv = engine.play(depth: 2);
  print("Engine plays: $mv");
}
