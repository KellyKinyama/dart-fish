import 'dart:math';
import '../engine/chess.dart';
import '../engine/chess_with_nnue.dart';
import '../engine/constants.dart';
import '../engine/nnue_reference.dart';
import '../engine/tt.dart';

// ================================================================
// CONSTANTS
// ================================================================

const double INF = 1e12;
const double FUTILITY_MARGIN = 150.0;
const int HISTORY_MAX = 1 << 20;

// ================================================================
// NNUE EVALUATION
// ================================================================

extension NNUEBoardEval on NNUERef {
  double evaluateBoard(List<Piece?> board, Color turn) {
    final acc = newAccumulator();
    refreshAccumulator(acc, board);
    return evaluate(acc, turn);
  }
}

// ================================================================
// ROOT MOVE CLASS
// ================================================================

class RootMove {
  final Move move;
  double score;
  RootMove(this.move, this.score);
}

// ================================================================
// MOVE HELPERS
// ================================================================

bool isQuiet(Move mv) =>
    (mv.flags & (BITS_CAPTURE | BITS_PROMOTION | BITS_EP_CAPTURE)) == 0;

bool squareInBoard(int sq) => sq >= 0 && sq < 64;

int captureScore(ChessWithNNUE pos, Move mv) {
  if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return 0;
  final victim = pos.board[mv.to];
  final attacker = pos.board[mv.from];
  if (victim == null || attacker == null) return 0;
  return (victim.type.shift + 1) * 10 - attacker.type.shift;
}

// ================================================================
// KILLER + HISTORY TABLES
// ================================================================

class SearchTables {
  final List<List<Move?>> killers = List.generate(
    128,
    (_) => <Move?>[null, null],
  );
  final history = List.generate(128, (_) => List.filled(64, 0));

  void addKiller(int ply, Move mv) {
    if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return;
    if (killers[ply][0] == mv) return;
    killers[ply][1] = killers[ply][0];
    killers[ply][0] = mv;
  }

  void addHistory(Color side, Move mv, int depth) {
    if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return;
    final s = side == Color.WHITE ? 0 : 1;
    final idx = s * 64 + mv.from;
    history[idx][mv.to] += depth * depth;

    if (history[idx][mv.to] > HISTORY_MAX) {
      for (var row in history) {
        for (int i = 0; i < row.length; i++) {
          row[i] >>= 1;
        }
      }
    }
  }

  int historyScore(Color side, Move mv) {
    if (!squareInBoard(mv.from) || !squareInBoard(mv.to)) return 0;
    final s = side == Color.WHITE ? 0 : 1;
    return history[s * 64 + mv.from][mv.to];
  }
}

// ================================================================
// STATIC EVALUATION (NO QUIESCENCE)
// ================================================================

double staticEval(ChessWithNNUE pos, int color) {
  return color * pos.nnue.evaluateBoard(pos.board, pos.turn);
}

// ================================================================
// NEGAMAX (NO QSEARCH) + FIXED TERMINAL LOGIC
// ================================================================

double negamax(
  ChessWithNNUE pos,
  int depth,
  double alpha,
  double beta,
  int color,
  TranspositionTable tt,
  SearchTables tables,
  int ply,
) {
  final hash = pos.zobristKey;

  // TT probe
  final entry = tt.probe(hash);
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

  // True terminal detection
  final moves = pos.generate_moves();

  if (moves.isEmpty) {
    if (pos.in_check) {
      // Checkmate against side to move
      return -1000000.0 * color;
    } else {
      return 0.0; // stalemate
    }
  }

  // Depth 0 after terminal handled
  if (depth == 0) {
    return staticEval(pos, color);
  }

  // MOVE ORDERING
  Move? hashMove = entry?.move;

  moves.sort((a, b) {
    if (!squareInBoard(a.from) || !squareInBoard(a.to)) return 1;
    if (!squareInBoard(b.from) || !squareInBoard(b.to)) return -1;

    if (a == hashMove) return -1;
    if (b == hashMove) return 1;

    final ac = !isQuiet(a);
    final bc = !isQuiet(b);

    if (ac && bc) return captureScore(pos, b) - captureScore(pos, a);
    if (ac) return -1;
    if (bc) return 1;

    int as = tables.historyScore(pos.turn, a);
    int bs = tables.historyScore(pos.turn, b);

    if (tables.killers[ply].contains(a)) as += 50000;
    if (tables.killers[ply].contains(b)) bs += 50000;

    return bs - as;
  });

  double best = -INF;
  Move? bestMove;
  int moveIndex = 0;

  for (final mv in moves) {
    pos.make_move(mv);

    // FUTILITY
    if (depth == 1 && isQuiet(mv) && !pos.in_check) {
      double eval = staticEval(pos, color);
      if (eval + FUTILITY_MARGIN <= alpha) {
        pos.undo_move();
        moveIndex++;
        continue;
      }
    }

    int newDepth = depth - 1;

    // LMR
    if (depth >= 3 && moveIndex > 0 && isQuiet(mv) && !pos.in_check) {
      newDepth = max(1, newDepth - 1);
    }

    double score;

    // PVS
    if (moveIndex == 0) {
      score = -negamax(
        pos,
        newDepth,
        -beta,
        -alpha,
        -color,
        tt,
        tables,
        ply + 1,
      );
    } else {
      score = -negamax(
        pos,
        newDepth,
        -alpha - 1,
        -alpha,
        -color,
        tt,
        tables,
        ply + 1,
      );

      if (score > alpha && score < beta) {
        score = -negamax(
          pos,
          newDepth,
          -beta,
          -alpha,
          -color,
          tt,
          tables,
          ply + 1,
        );
      }
    }

    pos.undo_move();

    if (score > best) {
      best = score;
      bestMove = mv;
    }

    if (score > alpha) {
      alpha = score;

      if (isQuiet(mv)) {
        tables.addHistory(pos.turn, mv, depth);
        tables.addKiller(ply, mv);
      }
    }

    if (alpha >= beta) break;

    moveIndex++;
  }

  Bound bound;
  if (best <= alpha) {
    bound = Bound.UPPER;
  } else if (best >= beta)
    bound = Bound.LOWER;
  else
    bound = Bound.EXACT;

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

// ================================================================
// ROOT SEARCH (ASPIRATION WINDOWS SAFE)
// ================================================================

RootMove rootSearch(
  ChessWithNNUE pos,
  int depth,
  TranspositionTable tt,
  SearchTables tables, {
  double? prevEval,
}) {
  double alpha = -INF;
  double beta = INF;

  if (prevEval != null) {
    double w = 500;
    alpha = prevEval - w;
    beta = prevEval + w;

    if (alpha >= beta) {
      alpha = -INF;
      beta = INF;
    }
  }

  for (;;) {
    double score = -negamax(pos, depth, -beta, -alpha, -1, tt, tables, 0);

    if (score <= alpha) {
      alpha -= 200;
      if (alpha < -1e11) {
        alpha = -INF;
        beta = INF;
      }
    } else if (score >= beta) {
      beta += 200;
      if (beta > 1e11) {
        alpha = -INF;
        beta = INF;
      }
    } else {
      for (final mv in pos.generate_moves()) {
        pos.make_move(mv);
        double s = -negamax(pos, depth - 1, -INF, INF, -1, tt, tables, 1);
        pos.undo_move();

        if (s == score) {
          return RootMove(mv, score);
        }
      }
    }
  }
}

// ================================================================
// ITERATIVE DEEPENING
// ================================================================

RootMove iterativeDeepening(ChessWithNNUE pos, int maxDepth) {
  final tt = TranspositionTable();
  final tables = SearchTables();

  RootMove? best;
  double? lastEval;

  for (int depth = 1; depth <= maxDepth; depth++) {
    best = rootSearch(pos, depth, tt, tables, prevEval: lastEval);
    lastEval = best.score;

    print(
      "info depth $depth score ${best.score} pv ${best.move.fromAlgebraic}${best.move.toAlgebraic}",
    );
  }

  return best!;
}

// ================================================================
// ENGINE WRAPPER
// ================================================================

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

    for (final m in generate_moves()) {
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
